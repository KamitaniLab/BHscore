import argparse
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

parser = argparse.ArgumentParser()
parser.add_argument('--prototxt_file', '-p', required=True)
opt = parser.parse_args()


net = caffe_pb2.NetParameter()

with open(opt.prototxt_file) as f:
    s = f.read()
    txtf.Merge(s, net)

layers = [l.name for l in net.layer]
rename_top_dict = {}
for i in range(1, len(layers)):
    l_bottom = net.layer[i-1]
    rename_top_dict[l_bottom.top[0]] = l_bottom.name
    l_bottom.top[0] = l_bottom.name

    assert len(l_bottom.top) == 1
        
    l = net.layer[i]
    for i in range(len(l.bottom)):
        l.bottom[i] = rename_top_dict[l.bottom[i]]


outfn = opt.prototxt_file.replace('prototxt', '_renamed_prototxt')
with open(outfn, 'w') as f:
    f.write(str(net))

