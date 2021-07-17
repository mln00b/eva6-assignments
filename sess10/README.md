# Assignment A

ResNet 18 trained on the TinyImageNet dataset: target validation accuracy - 50%

Augmentations Used:
1. Pad to `(72,72)`, random crop to `(64,64)`
2. Horizontal Flip - probability - `0.5`
3. CutOut - size `(32,32)`, probability - `0.5`

SGD Optimizer, trained using One Cycle LR
Max LR: 1.0
Trained for 10 epochs

Achieves `53.80% validation accuracy`, on the 10th Epoch.

# Assignment B

COCO Dataset:
`id`: class id. Total unique classes in the dataset - 80
`width`, `height`: width, height of the image
`bbox`: format - `[x, y, width, height]`, `x`,`y` being the top-left corner of the bounding box

Normalized the bounding boxes based on image's `width`, `height` to be in range of `0,1`
CSV sheet with normalized data: `sample_coco.csv`

KMeans on bounding box's `width`, `height` with cluster sizes: 3, 4, 5, 6

Anchor Boxes:


Anchors - 3 Clusters:  
```
[[0.28974134 0.28711913]
 [0.69289688 0.20134954]
 [0.20031795 0.69201832]]
```

Anchors - 4 Clusters:  
```
[[0.49395323 0.23465354]
 [0.20139058 0.74280724]
 [0.22350963 0.3259426 ]
 [0.99057558 0.15940826]]
```

Anchors - 5 Clusters:  
```
[[0.22548399 0.22692335]
 [0.15402617 0.99619036]
 [0.51352989 0.23484426]
 [1.00474685 0.15753439]
 [0.24406117 0.51136221]]
```

Anchors - 6 Clusters:
```  
[[0.5546443  0.17502765]
 [0.18121752 0.55018471]
 [0.22449003 0.22389558]
 [1.0917331  0.15414845]
 [0.15521382 1.06159694]
 [0.41654505 0.40994462]]
 ```