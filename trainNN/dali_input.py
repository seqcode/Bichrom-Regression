
from math import sqrt
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import math
import yaml

batch_size = 16

def tfrecord_pipeline(dspath, batch_size, num_threads, device="cpu", device_id=None,
                        shard_id=0, num_shards=1, reader_name="Reader",
                        seq=True, chroms=False, chroms_vlog=False, target=True, target_vlog=False, label=False, random_shuffle=True,
                        scaler_means=None, scaler_vars=None):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)

    feature_description = {}
    feature_description["seq"] = tfrec.VarLenFeature(tfrec.float32, -1.0)
    feature_description["label"] = tfrec.FixedLenFeature([], tfrec.int64, -1)
    feature_description["target"] = tfrec.FixedLenFeature([], tfrec.float32, -1.0)
    for ct in dspath["chromatin_tracks"]:
        feature_description[ct] = tfrec.VarLenFeature(tfrec.float32, -1.0)
    
    with pipe:
        inputs = fn.readers.tfrecord(
            name=reader_name,
            path=dspath['TFRecord'],
            index_path=dspath['TFRecord_idx'],
            features=feature_description,
            shard_id = shard_id,
            num_shards = num_shards,
            random_shuffle=random_shuffle,
            read_ahead=True, 
            prefetch_queue_depth=20,
            pad_last_batch=True)
        if device=="gpu":
            inputs['seq'] = inputs['seq'].gpu()
            for ct in dspath["chromatin_tracks"]: inputs[ct] = inputs[ct].gpu()
            inputs['target'] = inputs['target'].gpu()
            inputs['label'] = inputs['label'].gpu()
        seqdata = fn.expand_dims(inputs['seq'], axes=1, device=device)
        seqdata = fn.reshape(seqdata, shape=(4, -1), device=device)
        # normalize if provided with means/vars
        if scaler_means and scaler_vars:
            for index, ct in enumerate(dspath["chromatin_tracks"]): 
                inputs[ct] = (inputs[ct] - scaler_means[index])/sqrt(scaler_vars[index])
        chromsdata = fn.cat(*[fn.expand_dims(inputs[ct], axes=0, device=device) for ct in dspath["chromatin_tracks"]], axis=0, device=device)

        sample = []
        # seq
        if seq: sample.append(seqdata)
        # chroms
        if chroms: 
            # vlog if needed
            if chroms_vlog:
                sample.append(math.log(chromsdata + 1))
            else:
                sample.append(chromsdata)
        # target
        if target:
            if target_vlog: 
                sample.append(math.log(inputs['target'] + 1))
            else:
                sample.append(inputs['target'])
        # label
        if label: sample.append(inputs['label'])

        pipe.set_outputs(*sample)
    return pipe
