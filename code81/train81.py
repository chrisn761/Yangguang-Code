"""Fixed-epoch file-level CV over 81 repeated observations; not patient-independent CV."""
import argparse, hashlib, json, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from models import Model
from light_models import LightModel
from revised_models import revised

ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=ROOT/'data81pro'
RESULTS_DIR=ROOT/'results81_file10fold81proFinal'
NAMES=['GEL','GEL-no-attention-simple','GEL-no-residual-simple','GRU','LSTM','CNN-LSTM','MSCNN-LSTM','Transformer','GEL-CNN-attention','GEL-MSCNN-attention','GEL-unidirectional','GEL-no-attention-final','GEL-no-residual-final','LSTM-final']
STAGES=['IV','V','VI']

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def model_for(name):
    if name=='GEL': return Model('GEL')
    if name=='GEL-unidirectional': return revised(name)
    if name in ('GEL-no-attention-final','GEL-no-residual-final','LSTM-final'): return revised(name)
    if name=='GEL-CNN-attention': return LightModel('GEL-CNN-attention-compact')
    if name=='CNN-LSTM': return LightModel('CNN-LSTM-compact')
    if name not in NAMES: raise ValueError(name)
    return revised(name)

def reusable(d,a,f,splits):
    """Validate completed artifacts and exact split/data/config before reuse."""
    try:
        marker=json.loads((d/'COMPLETED.json').read_text())
        protocol=json.loads((d/'protocol.json').read_text())
        if marker.get('folds')!=10 or marker.get('files')!=81: return False
        if any(protocol.get(k)!=v for k,v in vars(a).items() if k!='force_retrain'): return False
        if protocol.get('standardization')!='inner-training files only': return False
        if protocol.get('selection')!='minimum inner-validation cross entropy; outer validation used once after selection': return False
        if protocol.get('data_directory')!=DATA_DIR.name or protocol.get('signal_strength')!=1.0 or protocol.get('weight_decay')!=.01 or protocol.get('label_smoothing')!=.1: return False
        current=model_for(a.model)
        old_arch=(d/'architecture.txt').read_text(encoding='utf-8')
        if old_arch!=str(current):
            # PyTorch versions differ in ModuleList repeated-block repr.
            import re
            def leaves(text):
                return {re.sub(r'^\([^)]*\):\s*','',line.strip()) for line in text.splitlines() if '=' in line}
            if a.model!='GEL' or protocol.get('parameters')!=sum(p.numel() for p in current.parameters()) or leaves(old_arch)!=leaves(str(current)): return False
        # Future versions require a per-model semantic identity. Legacy reuse
        # is allowed only for the three explicitly unchanged architectures.
        identity=model_identity(a.model)
        if 'model_identity' in protocol:
            if protocol['model_identity']!=identity: return False
        elif a.model not in ('GEL','CNN-LSTM','GEL-CNN-attention'): return False
        oof=pd.read_csv(d/'oof_predictions.csv')
        if len(oof)!=81 or oof.filename.nunique()!=81: return False
        for rel in ['pooled_file81/metrics.csv','pooled_file81/confusion_3class.csv','class_fold_metrics.csv','fold_metrics.csv','all_epochs_history.csv']:
            if not (d/rel).is_file(): return False
        for k,(dev,te) in enumerate(splits,1):
            tr,va=train_test_split(dev,test_size=.2,random_state=a.seed+k,stratify=f.iloc[dev].label_id)
            fd=d/f'fold_{k:02d}'
            for role,idx in [('train',tr),('inner_validation',va),('outer_validation',te)]:
                old=pd.read_csv(fd/(role+'_manifest.csv')).sort_values('filename')
                now=f.iloc[idx].sort_values('filename')
                for col in ['filename','sha256','label_id']:
                    if old[col].tolist()!=now[col].tolist(): return False
            if set(oof.loc[oof.fold==k,'filename'])!=set(f.iloc[te].filename): return False
            # Do not unpickle legacy checkpoints merely to decide reuse.
            # Their architecture/config/manifests are checked above.
            if not (fd/'selected_model.pth').is_file() or (fd/'selected_model.pth').stat().st_size==0: return False
        return True
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError):
        return False

def model_identity(name):
    import inspect
    # Per-model implementation identity, independent of unrelated entry scripts.
    m=model_for(name)
    forward=inspect.getsource(type(m).forward)
    return hashlib.sha256((str(m)+forward+'inner_ce_adamw001_cosine_epochs_v1').encode()).hexdigest()

def load():
    f=pd.read_csv(DATA_DIR/'metadata/file_mapping_81.csv').sort_values('filename').reset_index(drop=True)
    assert (f.signal_strength==1.0).all(), 'Expected simulated strength1.0 data81pro'
    assert json.loads((DATA_DIR/'metadata/generation_protocol.json').read_text())['coefficient']==1.0
    assert len(f)==81 and f.filename.nunique()==81
    assert f.simulated_instance_id.nunique()==27
    assert (f.groupby('simulated_instance_id').size()==3).all()
    assert (f.groupby('simulated_instance_id').label_id.nunique()==1).all()
    arrays=[]
    for r in f.itertuples():
        p=DATA_DIR/r.filename
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256
        x=pd.read_csv(p).to_numpy(dtype=np.float32)
        assert x.shape[1]==81 and len(x)>0 and np.isfinite(x).all()
        arrays.append(x)
    assert f.sha256.nunique()==81
    return f,arrays

def partition(f,s):
    result=[]
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=s)
    for tr,te in skf.split(np.zeros(len(f)),f.label_id):
        tr=np.asarray(tr); te=np.asarray(te)
        assert len(te)>0 and set(f.iloc[tr].label_id)=={0,1,2}
        # File-level protocol: repeated files from one simulated instance may
        # occur on both sides by design. The files themselves never overlap.
        assert not set(te)&set(tr)
        result.append((tr,te))
    assert sorted(np.concatenate([te for _,te in result]))==list(range(81))
    return result

def collate(items):
    x,y=zip(*items)
    return pad_sequence(x,batch_first=True),torch.tensor(y),torch.tensor([len(a) for a in x])

def loader(idx,arrays,f,mean,scale,batch,shuffle=False):
    data=[(torch.from_numpy(((arrays[i]-mean)/scale).astype(np.float32)),int(f.iloc[i].label_id)) for i in idx]
    order=np.random.permutation(len(data)).tolist() if shuffle else list(range(len(data)))
    chunks=[order[j:j+batch] for j in range(0,len(order),batch)]
    if shuffle and len(chunks)>1 and len(chunks[-1])==1: chunks[-2].extend(chunks.pop())
    return DataLoader(data,batch_sampler=chunks,collate_fn=collate)

def metrics(y,prob):
    y=np.asarray(y); pred=prob.argmax(1); cm=confusion_matrix(y,pred,labels=[0,1,2]); rows=[]
    for c,s in enumerate(STAGES):
        tp=int(cm[c,c]); support=int(cm[c].sum()); fp=int(cm[:,c].sum()-tp); fn=support-tp; tn=len(y)-tp-fp-fn
        recall=tp/support if support else np.nan
        precision=tp/(tp+fp) if support and tp+fp else (0. if support else np.nan)
        f1=2*tp/(2*tp+fp+fn) if support else np.nan
        auc=roc_auc_score(y==c,prob[:,c]) if 0<support<len(y) else np.nan
        # Per-class BA is binary one-vs-rest balanced accuracy, not recall.
        ba=(recall+tn/(tn+fp))/2 if support and tn+fp else np.nan
        rows.append(dict(stage=s,support=support,correct=tp,class_accuracy_recall=recall,accuracy=(tp+tn)/len(y),balanced_accuracy=ba,precision=precision,recall=recall,f1=f1,auc=auc,TP=tp,FN=fn,FP=fp,TN=tn))
    all_present=all(r['support']>0 for r in rows)
    met={'accuracy':float((pred==y).mean())}
    for name,key in [('balanced_accuracy','recall'),('macro_precision','precision'),('macro_recall','recall'),('macro_f1','f1'),('macro_auc','auc')]:
        met[name]=float(np.mean([r[key] for r in rows])) if all_present else np.nan
    met['balanced_accuracy_present_classes']=float(np.nanmean([r['recall'] for r in rows]))
    return met,rows,cm

def cm_save(cm,labels,path):
    pd.DataFrame(cm,index=labels,columns=labels).rename_axis('true/predicted').to_csv(path.with_suffix('.csv'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='Predicted',ylabel='True')
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(path.with_suffix('.png'),dpi=250); plt.close(fig)

def export(y,prob,dest):
    dest.mkdir(parents=True,exist_ok=True); met,rows,cm=metrics(y,prob)
    pd.DataFrame([met]).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(dest/'class_metrics.csv',index=False)
    cm_save(cm,STAGES,dest/'confusion_3class')
    for r in rows: cm_save(np.array([[r['TP'],r['FN']],[r['FP'],r['TN']]]),[r['stage'],'Other'],dest/('confusion_'+r['stage']+'_vs_rest'))
    return met,rows

def evaluate(model,dl,device):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for x,y,lengths in dl:
            z=model(x.to(device),lengths)
            assert torch.isfinite(z).all()
            ys.extend(y.tolist()); probs.append(z.softmax(1).cpu().numpy())
    return np.array(ys),np.concatenate(probs)

def run(a):
    assert a.epochs>0 and a.batch_size>=2
    f,arrays=load(); splits=partition(f,a.seed)
    print(f'[DATA] {DATA_DIR} | label-conditioned simulation strength1.0 | file-level 10-fold',flush=True)
    root=RESULTS_DIR/a.model; root.mkdir(parents=True,exist_ok=True)
    codehash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    sig=hashlib.sha256(json.dumps(dict(args=vars(a),hashes=f.sha256.tolist(),code=codehash),sort_keys=True).encode()).hexdigest()
    for d in root.glob('run_*'):
        marker=d/'COMPLETED.json'
        if not getattr(a,'force_retrain',False) and marker.exists() and reusable(d,a,f,splits):
            print('[SKIP verified data/splits/config/architecture]',a.model,d,flush=True); return
    out=root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); out.mkdir()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed(a.seed); example=model_for(a.model)
    (out/'architecture.txt').write_text(str(example),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(dict(**vars(a),signature=sig,parameters=sum(p.numel() for p in example.parameters()),data_directory=DATA_DIR.name,signal_strength=0.0,data_type='unenhanced preprocessed source files',unit='file_level_repeated_observation',selection='minimum inner-validation cross entropy; outer validation used once after selection',standardization='inner-training files only',noise=0,dropout=.3,weight_decay=.01,label_smoothing=.1,sequence_truncation=False,missing_class='strict 3class macro undefined; separate present-class BA',split='outer StratifiedKFold over 81 files; outer-development files split 80/20 stratified for inner validation; repeated files from one simulated instance may cross folds'),indent=2),encoding='utf-8'); del example
    planned=[]
    protocol_path=out/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    protocol['signal_strength']=1.0
    protocol['data_type']='label-conditioned artificial separability; not clinical evidence'
    protocol['generation_protocol']=json.loads((DATA_DIR/'metadata/generation_protocol.json').read_text())
    protocol['model_identity']=model_identity(a.model)
    protocol['architecture_note']='Final four-model revision: GEL-unidirectional = UniLSTM64 + residual blocks + attention; GEL-no-attention-final = same backbone/residual blocks with masked mean pooling; GEL-no-residual-final = same backbone/attention without residual addition; LSTM-final = independent one-layer UniLSTM64 baseline. These are structurally distinct.'
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for k,(tr,te) in enumerate(splits,1):
        for role,idx in [('train',tr),('validation',te)]: planned.append(f.iloc[idx].assign(fold=k,role=role))
    pd.concat(planned).to_csv(out/'all_splits.csv',index=False)
    preds=[]; foldrows=[]; classrows=[]; histories=[]; splitrows=[]
    for fold,(development,te) in enumerate(splits,1):
        seed(a.seed+fold); fd=out/f'fold_{fold:02d}'; fd.mkdir()
        inner_train,inner_val=train_test_split(development,test_size=.2,random_state=a.seed+fold,
                                               stratify=f.iloc[development].label_id)
        inner_train=np.asarray(inner_train); inner_val=np.asarray(inner_val)
        assert not set(inner_train)&set(inner_val)
        assert not set(development)&set(te)
        for role,idx in [('train',inner_train),('inner_validation',inner_val),('outer_validation',te)]:
            sub=f.iloc[idx].copy(); sub.to_csv(fd/(role+'_manifest.csv'),index=False)
            splitrows.append(sub.assign(fold=fold,role=role))
        overlap=sorted(set(f.iloc[development].original_patient_id)&set(f.iloc[te].original_patient_id))
        (fd/'original_source_overlap.json').write_text(json.dumps(overlap),encoding='utf-8')
        count=sum(len(arrays[i]) for i in inner_train)
        mean=sum(arrays[i].sum(0,dtype=np.float64) for i in inner_train)/count
        scale=np.sqrt(sum(((arrays[i].astype(np.float64)-mean)**2).sum(0) for i in inner_train)/count); scale[scale<1e-8]=1
        mean=mean.astype(np.float32); scale=scale.astype(np.float32); np.savez(fd/'scaler.npz',mean=mean,scale=scale)
        model=model_for(a.model).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
        train_lossfn=nn.CrossEntropyLoss(label_smoothing=.1); selection_lossfn=nn.CrossEntropyLoss(); history=[]; best=float('inf'); selected_epoch=None
        inner_loader=loader(inner_val,arrays,f,mean,scale,a.batch_size)
        print(f'[{a.model} FOLD {fold}/10] train={len(inner_train)} inner_val={len(inner_val)} outer_val={len(te)}',flush=True)
        for epoch in range(1,a.epochs+1):
            model.train(); total=0; train_y=[]; train_prob=[]
            for x,y,lengths in loader(inner_train,arrays,f,mean,scale,a.batch_size,True):
                opt.zero_grad(set_to_none=True); z=model(x.to(device),lengths); loss=train_lossfn(z,y.to(device))
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                # Online training predictions: dropout active, weights change across batches.
                # Reuse logits; no extra forward pass or validation-based selection.
                train_y.extend(y.tolist())
                train_prob.append(z.detach().softmax(1).cpu().numpy())
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step(); total+=loss.item()*len(y)
            tm,tc,_=metrics(np.asarray(train_y),np.concatenate(train_prob))
            iy,ip=evaluate(model,inner_loader,device)
            im,ic,_=metrics(iy,ip)
            with torch.no_grad():
                iloss=0.; ninner=0
                for x,y,lengths in inner_loader:
                    z=model(x.to(device),lengths); iloss+=selection_lossfn(z,y.to(device)).item()*len(y); ninner+=len(y)
            iloss/=ninner
            row=dict(model=a.model,fold=fold,epoch=epoch,train_loss=total/len(inner_train),inner_loss=iloss,lr=opt.param_groups[0]['lr'],
                     **{'train_'+k:v for k,v in tm.items()},**{'inner_'+k:v for k,v in im.items()})
            for cls in tc:
                row.update({'train_'+cls['stage']+'_'+k:v for k,v in cls.items() if k!='stage'})
            history.append(row)
            pd.DataFrame(history).to_csv(fd/'epoch_history.csv',index=False)
            pd.DataFrame(histories+history).to_csv(out/'all_epochs_history.csv',index=False)
            detail=' '.join(f'{k}={im[k]:.4f}' for k in ['accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','macro_auc'])
            print(f'[FOLD {fold} EPOCH {epoch}/{a.epochs}] train_loss={total/len(inner_train):.6f} inner_loss={iloss:.6f} INNER {detail} lr={row["lr"]:.8f}',flush=True)
            if iloss < best:
                best=iloss; selected_epoch=epoch
                torch.save(dict(model=model.state_dict(),name=a.model,epoch=epoch,inner_loss=iloss),fd/'selected_model.pth')
            scheduler.step()
        torch.save(dict(model=model.state_dict(),name=a.model,epoch=a.epochs),fd/'last_model.pth')
        checkpoint=torch.load(fd/'selected_model.pth',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model'])
        y,prob=evaluate(model,loader(te,arrays,f,mean,scale,a.batch_size),device)
        met,rows=export(y,prob,fd/'validation_results')
        foldrows.append(dict(fold=fold,n=len(te),selected_epoch=selected_epoch,selected_inner_loss=best,**met)); classrows.extend([dict(fold=fold,**r) for r in rows]); histories.extend(history)
        pred=f.iloc[te].copy(); pred['fold']=fold; pred['y_true']=y; pred['y_pred']=prob.argmax(1); pred['correct']=pred.y_true==pred.y_pred
        for c,s in enumerate(STAGES): pred['prob_'+s]=prob[:,c]
        pred.to_csv(fd/'predictions.csv',index=False); preds.append(pred)
        print(f'[FOLD {fold} DONE] selected_epoch={selected_epoch} outer_accuracy={met["accuracy"]:.4f} outer_balanced_accuracy={met["balanced_accuracy"]:.4f}',flush=True)
    df=pd.concat(preds).sort_values('filename'); assert len(df)==df.filename.nunique()==81
    df.to_csv(out/'oof_predictions.csv',index=False)
    export(df.y_true.to_numpy(),df[['prob_'+s for s in STAGES]].to_numpy(),out/'pooled_file81')
    units=df.groupby('simulated_instance_id').agg(y_true=('y_true','first'),**{'prob_'+s:('prob_'+s,'mean') for s in STAGES}).reset_index()
    units['y_pred']=units[['prob_'+s for s in STAGES]].to_numpy().argmax(1); units.to_csv(out/'instance27_predictions.csv',index=False)
    export(units.y_true.to_numpy(),units[['prob_'+s for s in STAGES]].to_numpy(),out/'pooled_instance27')
    fr=pd.DataFrame(foldrows); fr.to_csv(out/'fold_metrics.csv',index=False)
    fr.drop(columns=['fold','n']).agg(['mean','std','count']).T.to_csv(out/'fold_mean_std_valid_count.csv')
    cr=pd.DataFrame(classrows); cr.to_csv(out/'class_fold_metrics.csv',index=False)
    cr.groupby('stage')[['class_accuracy_recall','accuracy','balanced_accuracy','precision','recall','f1','auc']].agg(['mean','std','count']).to_csv(out/'class_fold_mean_std_valid_count.csv')
    pd.DataFrame(histories).to_csv(out/'all_epochs_history.csv',index=False)
    pd.concat(splitrows).to_csv(out/'all_splits.csv',index=False)
    (out/'COMPLETED.json').write_text(json.dumps(dict(signature=sig,folds=10,files=81,instances=27,selection='inner validation minimum cross entropy')),encoding='utf-8')
    print('[DONE]',out,flush=True)

def main(default='GEL'):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model',choices=NAMES,default=default)
    p.add_argument('--force-retrain',action='store_true',help='Create a fresh timestamp run even if matching completed results exist')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())

if __name__=='__main__': main()
