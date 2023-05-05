#!/usr/bin/env python
import numpy as np
from openvino.runtime import Core, Tensor, opset8, Type, Shape, PartialShape, Model, serialize, op, get_version
import time
import sys

def compare_fc(K, N, is_bias=False):
    fake_weight = np.arange(K*N).reshape([K,N]);
    weight = opset8.constant(fake_weight, Type.f32);
    param = opset8.parameter(PartialShape([1, -1, K]), Type.f32)
    matmul = opset8.matmul(param, weight, False, False)
    if is_bias:
        fake_bias = np.arange(N)
        bias = opset8.constant(fake_bias, Type.f32)
        add = opset8.add(matmul, bias)
        model = Model(add, [param], "fc")
    else:
        model = Model(matmul, [param], "fc")
    serialize(model, "ov_fc_{0}_{1}.xml".format(K, N), "ov_fc_{0}_{1}.bin".format(K, N))
    core = Core()
    compiled_model = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": "1"})
    m = [16]
    for i in m:
        fake_input = np.random.random(i* K).reshape([1, i, K]);
        request = compiled_model.create_infer_request()
        results = request.infer({
            compiled_model.inputs[0]: fake_input
        })
        print("test [{0}, {1}]".format(K, N))
        ref_np = np.matmul(fake_input, fake_weight)
        if is_bias:
            ref_np = np.add(ref_np, fake_bias)
        ov_result = next(iter(results.values()))
        result = np.isclose(ref_np, ov_result, rtol=1e-4, atol=1e-5)
        print("compare {0}".format(np.all(result)))
        print(ref_np.shape)
        if not np.all(result):
            print("np ref result {0}".format(ref_np))
            print("ov result {0}".format(ov_result))
        # print(results)


if __name__ == "__main__":
    print(get_version())
    test_cases = [[4864, 256], [256, 2048], [256,256], [256, 4233], [2048, 256]]
    # test_cases = [[64, 256]]
    for test in test_cases:
        compare_fc(test[0], test[1])
    for test in test_cases:
        compare_fc(test[0], test[1], True)
