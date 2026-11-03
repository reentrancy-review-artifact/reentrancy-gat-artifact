
import sys, os, json, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import warnings
mode = sys.argv[1]; D = '/content/drive/MyDrive/PROJECT'; DEV = torch.device('cuda')
nd_peringatan = []
if mode == 'deterministik':
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
warnings.showwarning = lambda m, c, f, l, file=None, line=None: nd_peringatan.append(str(m)[:90])

class M(nn.Module):
    def __init__(s, dim_input=22, h=512, k=2, p=0.5):
        super().__init__()
        s.gat1 = GATConv(dim_input, h // 8, heads=8, dropout=p); s.gat2 = GATConv(h, h // 4, heads=4, dropout=p)
        s.gat3 = GATConv(h, h // 4, heads=4, dropout=p); s.proj = nn.Linear(dim_input, h)
        s.n1 = nn.LayerNorm(h); s.n2 = nn.LayerNorm(h); s.n3 = nn.LayerNorm(h)
        s.cls = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, k))
        s.p = p
    def forward(s, d):
        h0 = s.proj(d.x); h1 = s.n1(F.elu(s.gat1(d.x, d.edge_index)) + h0); h1 = F.dropout(h1, p=s.p, training=s.training)
        h2 = s.n2(F.elu(s.gat2(h1, d.edge_index)) + h1); h2 = F.dropout(h2, p=s.p, training=s.training)
        h3 = s.n3(F.elu(s.gat3(h2, d.edge_index)) + h2)
        return s.cls(global_mean_pool(h3, d.batch))

@torch.no_grad()
def prob(m, data):
    m.eval(); out = []
    for b in DataLoader(data, batch_size=32):
        out += torch.softmax(m(b.to(DEV)), 1)[:, 1].cpu().tolist()
    return np.array(out)

o = torch.load(f'{D}/data/processed/pyg/v3_augmented_dedup.pt', weights_only=False)
tr_all, bench = o['train'], o['benchmark']
y = np.array([int(d.y) for d in tr_all]); g = np.array([d.nama.split('::')[0] for d in tr_all])
i_tr, i_va = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, g)))
tr = [tr_all[i] for i in i_tr]; va = [tr_all[i] for i in i_va]; yv = y[i_va]

def latih():
    torch.manual_seed(42)
    m = M(dim_input=o['dim']).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = np.array([int(d.y) for d in tr]); n0, n1 = (yt == 0).sum(), (yt == 1).sum()
    lf = nn.CrossEntropyLoss(weight=torch.tensor([len(yt) / (2 * n0), len(yt) / (2 * n1)], dtype=torch.float).to(DEV))
    ld = DataLoader(tr, batch_size=32, shuffle=True); best, bw, mandek, ep_b = -1, None, 0, 0
    for ep in range(1, 51):
        m.train()
        for b in ld:
            b = b.to(DEV); opt.zero_grad(); lf(m(b), b.y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        f1 = f1_score(yv, (prob(m, va) >= 0.5).astype(int), zero_division=0)
        if f1 > best: best, bw, mandek, ep_b = f1, copy.deepcopy(m.state_dict()), 0, ep
        else: mandek += 1
        if mandek >= 15: break
    m.load_state_dict(bw); return m, ep_b

mA, eA = latih(); mB, eB = latih()
pA, pB = prob(mA, va), prob(mB, va); bA, bB = prob(mA, bench), prob(mB, bench)
print(json.dumps({'mode': mode, 'epoch_terbaik': [eA, eB], 'maks_selisih_prob_val': float(np.abs(pA - pB).max()),
                  'prediksi_val_beda': int(((pA >= .5) != (pB >= .5)).sum()), 'n_val': len(va),
                  'maks_selisih_prob_bench': float(np.abs(bA - bB).max()),
                  'peringatan_nondeterministik': sorted(set(nd_peringatan))[:3]}))
