运行如下命令，编译本项目：
mkdir build && cd build && make all pdl
注：目前版本配置文件有硬编码，需按提示进行修改。

NCNN:
onnx转ncnn命令：
bin/ncnn/onnx2ncnn model_param/resnet18-v2-7.onnx model_param/ncnn/resnet18.param model_param/ncnn/resnet18.bin

运行推理命令：
bin/ncnn/resnet18_ncnn image/dog.jpg 

性能报告：
=== Performance Report (10 runs) ===
Min/Max/Avg Preprocessing Time: 0.10/0.11/0.10 ms
Min/Max/Avg Inference Time: 27.23/57.88/33.95 ms
Min/Max/Avg Postprocessing Time: 0.00/0.01/0.00 ms
Memory Usage: Base=224612 KB, Peak=235828 KB, Growth=11216 KB
Min/Max/Avg FPS: 17.24/36.58/29.36

Top 3 Results:
Class: 188, Score: 9.829282
Class: 218, Score: 8.137074
Class: 157, Score: 7.802757

MNN:
onnx转mnn命令：
./mnn/MNNConvert -f ONNX --modelFile ../model_param/resnet18-v2-7.onnx --MNNModel ../model_param/mnn/resnet18.mnn --bizCode biz

运行推理命令：
bin/mnn/resnet18_mnn model_param/mnn/resnet18.mnn image/dog.jpg 

性能报告：
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0
Input shape: [1, 3, 224, 224]
Initial memory: 264968 KB

=== Performance Report (10 runs) ===
Min/Max/Avg Preprocessing Time: 0.65/2.70/0.91 ms
Min/Max/Avg Inference Time: 19.64/38.58/22.75 ms
Min/Max/Avg Postprocessing Time: 0.01/0.01/0.01 ms
Memory Usage: Base=264968 KB, Peak=276900 KB, Growth=11932 KB
Min/Max/Avg FPS: 24.22/49.16/44.05

Top 3 Results:
Class: 188, Score: 9.5002
Class: 218, Score: 8.6247
Class: 215, Score: 8.5897

PaddleLite:
onnx转paddle内部格式命令：
x2paddle --framework=onnx --model=resnet18-v2-7.onnx --save_dir=pd_model
注：需先按照转换工具，转换后的文件保存至src/pdl/src/assets/models下

运行推理命令：
cd src/pdl/src/shell && ./run.sh resnet imagenet_224.txt test linux amd64 cpu

性能报告：
=== Performance Report (10 runs) ===
Min/Max/Avg Preprocessing Time: 0.73/5.46/1.41 ms
Min/Max/Avg Inference Time: 41.04/52.56/47.11 ms
Min/Max/Avg Postprocessing Time: 0.85/2.69/1.19 ms
Memory Usage: Base=178540 KB, Peak=195028 KB, Growth=16488 KB
Min/Max/Avg FPS: 16.47/23.46/20.12

Top 3 Results:
Class: 188, Score: 9.833639
Class: 218, Score: 8.139140
Class: 157, Score: 7.806087