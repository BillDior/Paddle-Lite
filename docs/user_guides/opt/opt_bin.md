## 使用opt转化模型

opt是 x86 平台上的可执行文件，需要在PC端运行：支持Linux终端和Mac终端。

### 帮助信息

执行opt时不加入任何输入选项，会输出帮助信息，提示当前支持的选项：

```bash
 ./opt
```
> **注意：** 如果您是通过[准备opt](../model_optimize_tool)页面中，"方法二：下载opt可执行文件" 中提供的链接下载得到的opt可执行文件，请先通过`chmod +x ./opt`命令为下载的opt文件添加可执行权限。

![](https://paddlelite-data.bj.bcebos.com/doc_images/1.png)

### 功能一：转化模型为Paddle-Lite格式
opt可以将PaddlePaddle的部署模型格式转化为Paddle-Lite 支持的模型格式，期间执行的操作包括：

- 将protobuf格式的模型文件转化为naive_buffer格式的模型文件，有效降低模型体积
- 执行“量化、子图融合、混合调度、Kernel优选”等图优化操作，提升其在Paddle-Lite上的运行速度、内存占用等效果

模型优化过程：

（1）准备待优化的PaddlePaddle模型

PaddlePaddle模型有两种保存格式：
   Combined Param：所有参数信息保存在单个文件`params`中，模型的拓扑信息保存在`__model__`文件中。

![opt_combined_model](https://paddlelite-data.bj.bcebos.com/doc_images%2Fcombined_model.png)

   Seperated Param：参数信息分开保存在多个参数文件中，模型的拓扑信息保存在`__model__`文件中。
![opt_seperated_model](https://paddlelite-data.bj.bcebos.com/doc_images%2Fseperated_model.png)

(2) 终端中执行`opt`优化模型
**使用示例**：转化`mobilenet_v1`模型

```shell
./opt --model_dir=./mobilenet_v1 \
      --valid_targets=arm \
      --optimize_out_type=naive_buffer \
      --optimize_out=mobilenet_v1_opt
```
以上命令可以将`mobilenet_v1`模型转化为arm硬件平台、naive_buffer格式的Paddle_Lite支持模型，优化后的模型文件为`mobilenet_v1_opt.nb`，转化结果如下图所示：

![opt_resulted_model](https://paddlelite-data.bj.bcebos.com/doc_images/2.png)


(3) **更详尽的转化命令**总结：

```shell
./opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86|x86_opencl|npu) \
    --record_tailoring_info =(true|false) \
    --quant_model=(true|false) \
    --quant_type=(QUANT_INT8|QUANT_INT16)
```

| 选项         | 说明 |
| ------------------- | ------------------------------------------------------------ |
| --model_dir         | 待优化的PaddlePaddle模型（非combined形式）的路径 |
| --model_file        | 待优化的PaddlePaddle模型（combined形式）的网络结构文件路径。 |
| --param_file        | 待优化的PaddlePaddle模型（combined形式）的权重文件路径。 |
| --optimize_out_type | 输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。 |
| --optimize_out      | 优化模型的输出路径。                                         |
| --valid_targets     | 指定模型可执行的backend，默认为arm。目前可支持x86、x86_opencl、arm、opencl、npu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为"npu,arm"。 |
| --record_tailoring_info | 当使用 [根据模型裁剪库文件](./library_tailoring.html) 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。 |
| --quant_model       | 设置是否使用opt中的动态离线量化功能。 |
| --quant_type        | 指定opt中动态离线量化功能的量化类型，可以设置为QUANT_INT8和QUANT_INT16，即分别量化为int8和int16。量化为int8对模型精度有一点影响，模型体积大概减小4倍。量化为int16对模型精度基本没有影响，模型体积大概减小2倍。|

* 如果待优化的fluid模型是非combined形式，请设置`--model_dir`，忽略`--model_file`和`--param_file`。
* 如果待优化的fluid模型是combined形式，请设置`--model_file`和`--param_file`，忽略`--model_dir`。
* `naive_buffer`的优化后模型为以`.nb`名称结尾的单个文件。
* `protobuf`的优化后模型为文件夹下的`model`和`params`两个文件。将`model`重命名为`__model__`用[Netron](https://lutzroeder.github.io/netron/)打开，即可查看优化后的模型结构。
* 删除`prefer_int8_kernel`的输入参数，`opt`自动判别是否是量化模型，进行相应的优化操作。
* `opt`中的动态离线量化功能和`PaddleSlim`中动态离线量化功能相同，`opt`提供该功能是为了用户方便使用。

### 功能二：统计模型算子信息、判断是否支持

opt可以统计并打印出model中的算子信息、判断Paddle-Lite是否支持该模型。并可以打印出当前Paddle-Lite的算子支持情况。

（1）使用opt统计模型中算子信息

下面命令可以打印出mobilenet_v1模型中包含的所有算子，并判断在硬件平台`valid_targets`下Paddle-Lite是否支持该模型

`./opt --print_model_ops=true  --model_dir=mobilenet_v1 --valid_targets=arm`

![opt_print_modelops](https://paddlelite-data.bj.bcebos.com/doc_images/3.png)

（2）使用opt打印当前Paddle-Lite支持的算子信息

`./opt --print_all_ops=true`

以上命令可以打印出当前Paddle-Lite支持的所有算子信息，包括OP的数量和每个OP支持哪些硬件平台：

![opt_print_allops](https://paddlelite-data.bj.bcebos.com/doc_images/4.png)

`./opt --print_supported_ops=true  --valid_targets=x86`

以上命令可以打印出当`valid_targets=x86`时Paddle-Lite支持的所有OP：

![opt_print_supportedops](https://paddlelite-data.bj.bcebos.com/doc_images/5.png)
