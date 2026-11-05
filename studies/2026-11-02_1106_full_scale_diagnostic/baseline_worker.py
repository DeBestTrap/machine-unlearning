#!/usr/bin/env python3
"""Full-pool SISA/ProtoSISA diagnostic with one timed deletion per original shard."""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import protosisa as proto
from container_storage import initialize_container, clear_models
from dataset_splits import fingerprint_cifar, load_manifest
from reporting import classification_evaluation, manifest, step_record, write_bundle
from reporting.core import write_json, runtime_provenance, hardware


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()


def signed_read(path):
    value=json.loads(Path(path).read_text())
    checksum=value.pop("payload_sha256")
    if digest(value)!=checksum:raise ValueError(f"Checksum mismatch: {path}")
    return value


def signed_write(path,value):write_json(path,{**value,"payload_sha256":digest(value)})


class FullLoader:
    def __init__(self,run,plan):
        from torchvision import datasets,transforms
        self.parent=load_manifest(run / "parent_split.json")
        development=json.loads((run / "development_split.json").read_text())
        if development["manifest_sha256"]!=plan["development_manifest_sha256"]:
            raise ValueError("Development membership changed")
        normalized=transforms.Normalize((.485,.456,.406),(.229,.224,.225))
        training=transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalized])
        evaluation=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),normalized])
        self.train_set=datasets.CIFAR10(root=plan["data_root"],train=True,download=False,transform=training)
        self.val_set=datasets.CIFAR10(root=plan["data_root"],train=True,download=False,transform=evaluation)
        if fingerprint_cifar(self.train_set.data,self.train_set.targets)!=self.parent["source_sha256"]:
            raise ValueError("Full-training source fingerprint changed")
        self.train_indices=np.asarray(self.parent["train_indices"])
        self.val_indices=np.asarray(development["confirmation_indices"])
        if len(self.train_indices)!=45000 or len(self.val_indices)!=1000 or set(self.train_indices)&set(self.val_indices):
            raise ValueError("Unexpected diagnostic pools")

    def load(self,indices,category="train"):
        if category=="train":dataset,source=self.train_set,self.train_indices
        elif category=="train_eval":dataset,source=self.val_set,self.train_indices
        elif category=="val":dataset,source=self.val_set,self.val_indices
        else:raise ValueError("Only full training and fixed confirmation may be accessed")
        values=[dataset[int(source[index])] for index in indices]
        return torch.stack([x[0] for x in values]).numpy(),np.asarray([x[1] for x in values])


def requests_for_shards(shards,seed):
    rng=random.Random(seed)
    return [dict(shard=shard_id,local_index=int(rng.choice(list(shard)))) for shard_id,shard in enumerate(shards)]


def evaluate(directory,config,paths,prototypes,step,loader,encoder,device):
    evaluation=proto.evaluate(directory,config,paths,prototypes,step,loader,encoder,device,model_factory=proto.build_model)
    arrays=np.load(directory / "outputs" / f"components-step-{step}-val.npz")
    return dict(sisa=float(np.mean(arrays["ensemble"].argmax(1)==arrays["labels"]))*100,
                protosisa=float(evaluation["metrics"]["accuracy"])*100),evaluation,arrays["ensemble"].copy(),arrays["labels"].copy()


def clean_unfinished_step(directory,config,shard_id,shard,deleted,step):
    """Redo an uncommitted measurement from the original unaffected prefix.

    Every original shard is requested only once. Original training checkpoints
    remain immutable; only this request's new checkpoints/statistics are removed.
    """
    original={path for path,_,_ in proto.checkpoint_paths(directory,config,shard_id,shard,[])}
    changed=proto.checkpoint_paths(directory,config,shard_id,shard,deleted)
    for path,_,_ in changed:
        if path not in original:path.unlink(missing_ok=True)
    for name in (f"shard-{shard_id}:{step}.json",f"shard-{shard_id}:{step}.time",f"prototypes:{step}.json"):
        path=directory / "times" / name
        if path.exists():
            archive=directory / "timing_retries"
            archive.mkdir(exist_ok=True)
            import time
            path.replace(archive / f"{time.time_ns()}-{path.name}")
    (directory / "cache" / f"prototypes-step-{step}.pt").unlink(missing_ok=True)
    indices=proto.retained_indices(shard,deleted)
    key=proto.digest([config["training_sha256"],shard_id,indices.tolist()])
    (directory / "cache" / f"prototype-{key}.pt").unlink(missing_ok=True)


def publish_reports(directory,config,plan,job,initial,final,initial_evaluation,final_evaluation,ensemble,labels,events,device):
    total=sum(len(shard) for shard in np.load(directory / "splitfile.npy",allow_pickle=True))
    sizes=list(map(len,np.load(directory / "splitfile.npy",allow_pickle=True)))
    methods={}
    for method in ("sisa","protosisa"):
        times=[row[f"{method}_update_sec"] for row in events]
        methods[method]=dict(initial_accuracy=initial[method],final_accuracy=final[method],
            mean_update_sec=statistics.mean(times),std_update_sec=statistics.stdev(times) if len(times)>1 else 0.,
            weighted_mean_update_sec=sum(row[f"{method}_update_sec"]*sizes[row["shard"]]/total for row in events))
        run_id=f"full-scale-{job['id']}-{method}"
        report=directory / method
        evaluations=[]
        for step,evaluation in ((0,initial_evaluation),(len(events),final_evaluation)):
            if method=="protosisa":
                value=dict(evaluation,run_id=run_id)
                # Prediction artifacts are relative to the parent record folder.
                source=directory / value["prediction_artifact"]
                target=report / value["prediction_artifact"]
                target.parent.mkdir(parents=True,exist_ok=True)
                import shutil
                shutil.copy2(source,target)
            else:
                arrays=np.load(directory / "outputs" / f"components-step-{step}-val.npz")
                value=classification_evaluation(report,run_id=run_id,state_id=f"step-{step}",step=step,
                    split="val",dataset="cifar10",labels=arrays["labels"],predictions=arrays["ensemble"].argmax(1),
                    scores=arrays["ensemble"],num_classes=10,split_sha256=plan["parent_manifest_sha256"])
            value["method_details"].update(evaluation_pool="confirmation",evaluation_source_indices_sha256=config["evaluation_source_indices_sha256"],
                selection_feedback=False,diagnostic_only=True)
            evaluations.append(value)
        steps=[step_record(run_id,0)]
        for event in events:
            steps.append(step_record(run_id,event["step"],deleted_count=1,deleted_total=event["step"],
                source_ids=[event["source_index"]],update_time_sec=event[f"{method}_update_sec"],
                timing_scope="update_v1",method_details=dict(shard=event["shard"],diagnostic_only=True)))
        write_bundle(report,manifest(run_id=run_id,method=method,dataset="cifar10",model="resnet50",config=config,
            split_sha256=plan["parent_manifest_sha256"],requested_steps=len(events),seed=plan["seed"],
            provenance=runtime_provenance(ROOT),hardware_info=hardware(device),precision="fp32"),steps,evaluations)
    return methods


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--job",required=True)
    parser.add_argument("--stage",choices=("train","replay"),required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args(argv)
    run=args.run.resolve();plan=signed_read(run / "plan.json")
    if any(file_hash(path)!=checksum for path,checksum in plan["sources"].items()):raise ValueError("Source changed since preparation")
    job=next(x for x in plan["jobs"] if x["id"]==args.job)
    records=Path(job["records"]);records.mkdir(parents=True,exist_ok=True)
    marker=records / f"{args.stage}.json"
    if marker.exists():
        value=signed_read(marker)
        if value["plan_sha256"]!=file_hash(run / "plan.json"):
            raise ValueError("Saved baseline result belongs to another plan")
        if args.stage=="replay" and not value.get("checkpoints_discarded"):
            clear_models([job["baseline_records"]])
            value["checkpoints_discarded"]=True
            signed_write(marker,value)
        return
    directory=initialize_container(job["baseline_records"])
    torch.set_num_threads(plan["threads"])
    device=torch.device(args.device)
    if device.type=="cuda":torch.cuda.set_device(device)
    loader=FullLoader(run,plan)
    config=dict(model="resnet50",dataset=str(run / "inputs/baseline_datasetfile"),
        dataset_metadata=dict(nb_train=plan["training_examples"],nb_val=plan["evaluation_examples"],nb_test=0,nb_classes=10,input_shape=[32,32]),
        seed=plan["seed"],shards=job["shards"],slices=job["slices"],epochs=job["epochs"],
        learning_rate=job["lr"],optimizer="adamw",batch_size=32,dropout_rate=0.,
        prototype_weighting="safe",prototype_weight=None,eval_split="val",
        comparison_run_id=f"full-scale-{job['id']}",plan_sha256=file_hash(run / "plan.json"),
        evaluation_source_indices_sha256=digest(loader.val_indices.tolist()))
    config["training_sha256"]=digest(config)
    config_path=directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError("Frozen baseline config changed")
    write_json(config_path,config)
    write_json(directory / "dataset_split.json",loader.parent)
    write_json(directory / "diagnostic_evaluation.json",dict(source_indices=loader.val_indices.tolist(),pool="confirmation",selection_feedback=False))
    shards=np.array_split(np.random.default_rng(plan["seed"]).permutation(plan["training_examples"]),job["shards"])
    array=np.empty(len(shards),dtype=object);array[:]=shards
    np.save(directory / "splitfile.npy",array,allow_pickle=True)
    encoder=proto.build_encoder("resnet50",device)
    if args.stage=="train":
        paths=[proto.train_member(directory,config,i,shard,[],0,loader,device,model_factory=proto.build_model) for i,shard in enumerate(shards)]
        prototypes=proto.update_prototypes(directory,config,shards,[[] for _ in shards],0,loader,encoder,device)
        initial,evaluation,_,_=evaluate(directory,config,paths,prototypes,0,loader,encoder,device)
        value=dict(plan_sha256=file_hash(run / "plan.json"),job=job["id"],baseline_records=str(directory),
            member_paths=list(map(str,paths)),initial_accuracy=initial,training_examples=plan["training_examples"],evaluation_examples=plan["evaluation_examples"],
            checkpoint_retention="until_shard_sweep_complete")
        write_json(directory / "initial_evaluation.json",evaluation)
        signed_write(marker,value)
        print(json.dumps(value,indent=2),flush=True)
        return
    trained=signed_read(records / "train.json")
    paths=[Path(x) for x in trained["member_paths"]]
    deleted=[[] for _ in shards]
    events=[]
    prototypes=proto.update_prototypes(directory,config,shards,deleted,0,loader,encoder,device)
    for step,request in enumerate(requests_for_shards(shards,plan["deletion_seed"]),1):
        shard_id,local_index=request["shard"],request["local_index"]
        deleted[shard_id].append(local_index)
        step_path=directory / "times" / f"diagnostic-step-{step}.json"
        if step_path.exists():
            event=signed_read(step_path)
            if event["local_index"]!=local_index or event["plan_sha256"]!=file_hash(run / "plan.json"):
                raise ValueError("Saved shard request changed")
            paths=[Path(x) for x in event["member_paths"]]
            prototypes=torch.load(directory / "cache" / f"prototypes-step-{step}.pt",map_location="cpu",weights_only=True)
            events.append(event);continue
        clean_unfinished_step(directory,config,shard_id,shards[shard_id],deleted[shard_id],step)
        start=proto.synchronized_time(device)
        paths[shard_id]=proto.train_member(directory,config,shard_id,shards[shard_id],deleted[shard_id],step,loader,device,model_factory=proto.build_model)
        sisa_seconds=proto.synchronized_time(device)-start
        start=proto.synchronized_time(device)
        prototypes=proto.update_prototypes(directory,config,shards,deleted,step,loader,encoder,device)
        proto.mixture_parameters(config,plan["training_examples"]-step,len(shards))
        prototype_seconds=proto.synchronized_time(device)-start
        event=dict(step=step,shard=shard_id,local_index=local_index,source_index=int(loader.train_indices[local_index]),
            sisa_update_sec=sisa_seconds,protosisa_update_sec=sisa_seconds+prototype_seconds,
            prototype_maintenance_sec=prototype_seconds,plan_sha256=file_hash(run / "plan.json"),member_paths=list(map(str,paths)))
        signed_write(step_path,event);events.append(event)
        print(json.dumps(event),flush=True)
    if sorted(e["shard"] for e in events)!=list(range(len(shards))):raise ValueError("Incomplete original-shard coverage")
    final,evaluation,ensemble,labels=evaluate(directory,config,paths,prototypes,len(shards),loader,encoder,device)
    initial_evaluation=json.loads((directory / "initial_evaluation.json").read_text())
    methods=publish_reports(directory,config,plan,job,trained["initial_accuracy"],final,initial_evaluation,evaluation,ensemble,labels,events,device)
    value=dict(plan_sha256=file_hash(run / "plan.json"),job=job["id"],requests=len(events),methods=methods,
        baseline_records=str(directory),training_examples=plan["training_examples"],evaluation_examples=plan["evaluation_examples"],
        checkpoints_discarded=False,timing_scope="update_v1",shard_sizes=list(map(len,shards)))
    signed_write(marker,value)
    clear_models([str(directory)])
    value["checkpoints_discarded"]=True
    signed_write(marker,value)
    print(json.dumps(value,indent=2),flush=True)


if __name__=="__main__":main()
