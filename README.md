# ComfyUI-Auvart

Auvart (Hidden Memory Art) icin deneysel ComfyUI custom node'lari ve yardimci scriptler.

## Icerik

- **ComfyUI-Auvart/** — ComfyUI custom node paketi:
  - `Auvart Photo Preprocessor` — foto on-isleme
  - `Auvart Hybrid Image` — hibrit gorsel uretimi
  - `Auvart FFT Phase Transfer` — FFT faz transferi
  - `Auvart Photo To Mask` — fotodan maske cikarma
  - `Auvart Squint Simulator` — gizli gorsel etkisini onizleme
- `auvart_toolkit.py` — bagimsiz yardimci toolkit (ComfyUI gerektirmez)
- `batch_tester.py` — toplu test scripti
- `requirements.txt` — Python bagimliliklari

## Kurulum

### 1. ComfyUI kur (yoksa)

Windows icin en kolayi portable surum: https://github.com/comfyanonymous/ComfyUI/releases
adresinden `ComfyUI_windows_portable` zip'ini indir, bir klasore cikar,
`run_nvidia_gpu.bat` ile calistir (NVIDIA ekran karti gerekir, yoksa `run_cpu.bat`).

### 2. Bu node'lari ekle

Bu repo'yu ComfyUI'nin `custom_nodes` klasorune klonla:

```
cd ComfyUI/custom_nodes        # portable surumde: ComfyUI_windows_portable/ComfyUI/custom_nodes
git clone https://github.com/sefakocakalay/auvart-comfyui.git
```

### 3. Bagimliliklari yukle

```
pip install -r auvart-comfyui/requirements.txt
```

Portable surum kullaniyorsan pip yerine:

```
ComfyUI_windows_portable/python_embeded/python.exe -m pip install -r requirements.txt
```

### 4. ComfyUI'yi yeniden baslat

Node'lar sag tik menusunde **Auvart** kategorisi altinda gorunur.

## Bagimsiz scriptler

`auvart_toolkit.py` ve `batch_tester.py` ComfyUI olmadan da calisir:

```
pip install -r requirements.txt
python auvart_toolkit.py --help
```
