# DiabeticRetinopathy
In this repository we create a model to compete in this [Kaggle competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) 8 years after it has closed.

The results will be published in a blog post...


To install this repo run the following commands:

```
git clone <https-or-ssh>
```

Required packages are:

```
PyTorch >= 1.7.0
torchvision >= 0.8.1
timm == 0.3.2 
```
For `timm` a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) (also add `import torch` to this code) is needed to work with PyTorch 1.8.1+.

Download the weights for the ViT and put them in the folder `weights`. (Credits to this [page](https://github.com/facebookresearch/mae/blob/main/README.md?plain=1))
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

```
Some command
```

Download the training data from [Kaggle](#) and put it in the folder `data`.