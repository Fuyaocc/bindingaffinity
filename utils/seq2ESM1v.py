#生成esm1v的信息
import esm,torch
import logging
def seq2ESM1v(seq, model, alphabet, batch_converter, device):
    res=[]
    data = [("tmp", seq),]
    _, _, batch_tokens = batch_converter(data)
    for i in range(len(seq)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0][i+1]=alphabet.mask_idx  #mask the residue,32
        # print(batch_tokens_masked)
        with torch.no_grad():
            x=model(batch_tokens_masked.to(device))
        res.append(x[0][i+1].tolist())
    
    res=torch.Tensor(res).to(device)
    # print(res)
    return res

    
