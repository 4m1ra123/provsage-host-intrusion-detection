import json
import pandas as pd
import torch
import numpy as np


def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)


def add_attributes(df, json_file_path):
    df = df.astype(str)
    wanted_keys = set(zip(df['actorID'], df['objectID'], df['action'], df['timestamp']))

    info = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "EVENT" not in line:
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            event = x.get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event', {})

            action = event.get('type', '')
            actor = event.get('subject', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
            obj_raw = event.get('predicateObject') or {}
            obj = obj_raw.get('com.bbn.tc.schema.avro.cdm18.UUID', '') if obj_raw else ''
            timestamp = str(event.get('timestampNanos', ''))
            cmd = event.get('properties', {}).get('map', {}).get('exec', '')

            path_v = ''
            if event.get('predicateObjectPath') is not None:
                path_v = event['predicateObjectPath'].get('string', '')

            path2 = ''
            if event.get('predicateObject2Path') is not None:
                path2 = event['predicateObject2Path'].get('string', '')

            obj2 = ''
            if event.get('predicateObject2') is not None:
                obj2 = event['predicateObject2'].get('com.bbn.tc.schema.avro.cdm18.UUID', '')

            actor_s = str(actor)

            if obj2 and (actor_s, str(obj2), action, timestamp) in wanted_keys:
                info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path2})

            if obj and (actor_s, str(obj), action, timestamp) in wanted_keys:
                info.append({'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path_v})

    rdf = pd.DataFrame.from_records(info).astype(str)
    return df.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()

def prepare_graph(df):
    nodes, labels, edges, edge_attr = {}, {}, [], []
    dummies = {
        "SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_CHAR": 2, "FILE_OBJECT_FILE": 3,
        "FILE_OBJECT_DIR": 4, "SUBJECT_UNIT": 5, "UnnamedPipeObject": 6, "FILE_OBJECT_UNIX_SOCKET": 7,
        "SRCSINK_UNKNOWN": 8, "FILE_OBJECT_LINK": 9, "NetFlowObject": 10, "FILE_OBJECT_BLOCK": 11
    }

    num_classes = len(dummies)

    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if 'path' in row and row['path'] else [])

        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = dummies[row['actor_type']]

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = dummies[row['object']]

        edges.append((actor_id, object_id))
        edge_attr.append(float(row["timestamp"]))

    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    one_hot_labels = []

    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

        # 生成单热编码向量
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[labels[node_id]] = 1
        one_hot_labels.append(one_hot_vector)


    update_edge_index(edges, edge_index, index_map)

    return features, feat_labels, edge_index, list(index_map.keys()), one_hot_labels, edge_attr


