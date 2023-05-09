python affinity_gcn.py --device cuda:0 --epoch 2500 --dim 80 --batch_size 64 --alpha 0.5
python affinity_test.py --device cuda:0 --dim 80 --batch_size 32

python affinity_gcn.py --device cuda:0 --epoch 2500 --dim 80 --batch_size 64 --alpha 0.3
python affinity_test.py --device cuda:0 --dim 80 --batch_size 32