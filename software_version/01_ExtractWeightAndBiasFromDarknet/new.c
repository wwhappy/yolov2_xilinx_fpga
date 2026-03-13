void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        // 如是二值卷积，可专门处理，这里忽略
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    // 计算所有卷积核参数总数量（适配groups分组卷积做通道约束）
    int num = l.c/l.groups * l.n * l.size * l.size;

    // 读取偏置
    fread(l.biases, sizeof(float), l.n, fp);

    // 如果有BN且允许读scale，则读取BN对应参数
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);             // BN参数 scale/gamma
        fread(l.rolling_mean, sizeof(float), l.n, fp);       // BN参数 均值
        fread(l.rolling_variance, sizeof(float), l.n, fp);   // BN参数 方差
    }
    // 读取卷积核权重
    fread(l.weights, sizeof(float), num, fp);

    // 如需转换权重格式，则转置矩阵
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }

#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif

    ///add start
    int i,j;
    FILE *fp_w = fopen("weights.bin", "ab+");   // 写融合后权重
    if(!fp_w) file_error("weights.bin");
    FILE *fp_bias = fopen("bias.bin", "ab+");   // 写融合后偏置
    if(!fp_bias) file_error("bias.bin");

    float *weight_buffer = (float *)calloc(num, sizeof(float)); // 存储融合后权重
    float *alpha_buffer = (float *)calloc(l.n, sizeof(float));  // 每个输出通道的BN融合系数 alpha
    float *bias_buffer = (float *)calloc(l.n, sizeof(float));   // 每个输出通道的BN融合后偏置

    // 计算融合参数
    if(l.batch_normalize && (!l.dontloadscales))
    {   // 有BN时, 按BN融合公式合并scale/均值/方差等到权重与偏置
        for(i = 0;i < l.n; i++)
        {
            float tmp = l.scales[i]/(sqrt(l.rolling_variance[i]) + .000001f); // alpha = scale / sqrt(variance + ε)
            alpha_buffer[i] = tmp;
            bias_buffer[i] = l.biases[i] - l.rolling_mean[i]*tmp;             // beta = bias - mean * alpha
        }
    }
    else
    {   // 无BN，保持原样
        for(i = 0;i < l.n; i++)
        {
            alpha_buffer[i] = 1;
            bias_buffer[i] = l.biases[i];
        }
    }

    // 合并权重（每个输出通道j，对应的所有参数乘以alpha[j]）
    int cnt = 0;
    for(j = 0;j < l.n; j++)
        for(i = 0; i < num/l.n; i++)
        {
            weight_buffer[cnt] = l.weights[cnt]*alpha_buffer[j];
            cnt++;
        }
    // 写融合权重与融合偏置（分别存为bin文件,便于后端加载和FPGA解析）
    fwrite(weight_buffer, sizeof(float), num, fp_w);
    fwrite(bias_buffer, sizeof(float), l.n, fp_bias);

    // 清理
    fclose(fp_w);
    fclose(fp_bias);
    free(weight_buffer);
    free(alpha_buffer);
    free(bias_buffer);    
    ///add end 
}