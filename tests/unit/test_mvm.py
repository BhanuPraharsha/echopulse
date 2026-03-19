import pytest, torch
from mvm.masking.phase_aware_mask import PhaseAwareMask
from mvm.encoder.transformer_encoder import CardiacTransformerEncoder
from mvm.decoder.multi_task_decoder import MultiTaskDecoder
from mvm.loss import MVMLoss

@pytest.fixture
def vol(): return torch.randn(8,32,112,112)

@pytest.fixture
def small_enc():
    return CardiacTransformerEncoder(embed_dim=128,depth=2,num_heads=4)

@pytest.fixture
def small_dec():
    return MultiTaskDecoder(embed_dim=128)

def test_mask_ratio(vol):
    mv,m = PhaseAwareMask(0.5)(vol[0])
    assert abs(m.float().mean().item()-0.5) < 0.15

def test_masked_zero(vol):
    mv,m = PhaseAwareMask(0.5)(vol[0])
    assert mv[m].abs().max()==0.0

def test_encoder_shape(vol,small_enc):
    ph = torch.zeros(8,32,dtype=torch.long)
    out = small_enc(vol,ph)
    assert out.shape==(8,32*49,128)

def test_decoder_shapes(vol,small_enc,small_dec):
    ph=torch.zeros(8,32,dtype=torch.long)
    enc_out=small_enc(vol,ph)
    r,ef,p=small_dec(enc_out,T=32)
    assert r.shape==(8,32,49,256)
    assert ef.shape==(8,1)
    assert p.shape==(8,32,2)

def test_loss_positive():
    lf=MVMLoss()
    B,T,P,px=4,16,49,256
    rp=torch.rand(B,T,P,px); rt=torch.rand(B,T,P,px)
    m=torch.zeros(B,T,dtype=torch.bool); m[:,:8]=True
    pp=torch.rand(B,T,2); pt=torch.randint(0,2,(B,T))
    ep=torch.rand(B,1); et=torch.rand(B,1)
    _,L=lf(rp,rt,m,pp,pt,ep,et)
    assert L.item()>0