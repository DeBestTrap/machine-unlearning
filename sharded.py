import numpy as np
from hashlib import sha256
import importlib  # Keep the import hook used by cross-repo deletion replay audits.
import json
from dataset_loading import load_dataset_module
from split_support import validate_container_split
from container_storage import record_directory

def sizeOfShard(container, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    shards = np.load(record_directory(container) / 'splitfile.npy', allow_pickle=True)
    
    return shards[shard].shape[0]

def realSizeOfShard(container, label, shard):
    '''
    Returns the actual size of the shard (including unlearning requests).
    '''
    shards = np.load(record_directory(container) / 'splitfile.npy', allow_pickle=True)
    requests = np.load(record_directory(container) / f'requestfile:{label}.npy', allow_pickle=True)
    
    return shards[shard].shape[0] - requests[shard].shape[0]

def getShardHash(container, label, shard, until=None):
    '''
    Returns a hash of the indices of the points in the shard lower than until
    that are not in the requests (separated by :).
    '''
    shards = np.load(record_directory(container) / 'splitfile.npy', allow_pickle=True)
    requests = np.load(record_directory(container) / f'requestfile:{label}.npy', allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard])
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(container, label, shard, batch_size, dataset, offset=0, until=None):
    '''
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionnally located between offset and until (slicing).
    '''
    validate_container_split(container, dataset)
    shards = np.load(record_directory(container) / 'splitfile.npy', allow_pickle=True)
    requests = np.load(record_directory(container) / f'requestfile:{label}.npy', allow_pickle=True)
    
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = load_dataset_module(dataset, datasetfile)
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard])
        yield dataloader.load(indices)
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        yield dataloader.load(indices)

def fetchTestBatch(dataset, batch_size, category="test"):
    '''
    Generator returning batches of points from the specified test dataset
    with specified batch_size.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = load_dataset_module(dataset, datasetfile)

    if category not in ("test", "val"):
        raise ValueError("Evaluation category must be test or val")
    size = datasetfile[f"nb_{category}"]
    limit = 0
    while limit <= size - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), category=category)
    if limit < size:
        yield dataloader.load(np.arange(limit, size), category=category)
