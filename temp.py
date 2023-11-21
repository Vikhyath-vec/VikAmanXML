import numpy as np

path = "/home/es20btech11035/VikAmanXML/knowledge_graph_attention_network/Data/yelp2018/item_list.txt"
org_id_remap_id = {}
with open(path, 'r') as file:
    next(file)
    for line in file:
        org_id, remap_id = line.strip().split()
        org_id_remap_id[org_id] = int(remap_id[:-len(org_id)])

print(org_id_remap_id['3fw2X5bZYeW9xCz_zGhOHg'])

# Assuming filename is the path to your .npz file
filename = '/home/es20btech11035/VikAmanXML/knowledge_graph_attention_network/Model/pretrain/yelp2018/kgat.npz'

# # Load the data from the .npz file
data = np.load(filename)

item_embed = data['entity_embed']
print(item_embed[org_id_remap_id['3fw2X5bZYeW9xCz_zGhOHg']].shape)
# item_embed = data['entity_embed']
# relation_embed = data['relation_embed']

# u = user_embed[0]
# i = item_embed[0]
# # print(u)
# print(user_embed.shape)
# print(item_embed.shape)
# print(relation_embed.shape)

# print(u.shape)
# print(i.shape)

# print(np.dot(u,i))


# print(u.shape)
# print(i.shape)

# print(user_embed.shape)
# print(np.outer(i,u))

