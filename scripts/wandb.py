import wandb                                                                                                                     
from wandb.proto import wandb_internal_pb2 as pb                                                                                 
from google.protobuf.json_format import MessageToDict                                                                            
import matplotlib.pyplot as plt
from collections import defaultdict                                                                                              
                
path = "/mnt/data/szf_temp/SpecForge/wandb/wandb/offline-run-20260416_101101-dohbz7f6/run-dohbz7f6.wandb"                        

history = defaultdict(list)                                                                                                      
                
with open(path, "rb") as f:
    while True:
        # 每条记录前4字节是长度
        header = f.read(4)                                                                                                       
        if len(header) < 4:
            break                                                                                                                
        length = int.from_bytes(header, "little")
        data = f.read(length)                                                                                                    
        if len(data) < length:
            break                                                                                                                

        record = pb.Record()                                                                                                     
        try:    
            record.ParseFromString(data)
        except Exception:
            continue                                                                                                             

        if record.HasField("history"):                                                                                           
            d = MessageToDict(record.history)
            for item in d.get("item", []):
                k = item.get("key")                                                                                              
                v = item.get("valueJson")
                if k and v:                                                                                                      
                    try:                                                                                                         
                        history[k].append(float(v))
                    except:                                                                                                      
                        pass

# 画图
fig, axes = plt.subplots(len(history), 1, figsize=(10, 3 * len(history)))
if len(history) == 1:                                                                                                            
    axes = [axes]
for ax, (k, vals) in zip(axes, history.items()):                                                                                 
    ax.plot(vals)
    ax.set_title(k)                                                                                                              
    ax.set_xlabel("step")
plt.tight_layout()                                                                                                               
plt.savefig("training_curves.png", dpi=150)
print("Saved to training_curves.png")                                                                                            
print("Keys found:", list(history.keys()))