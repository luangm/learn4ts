import Learn from "../../src/index";
import {Tensor} from "tensor4js";
import Learn4js from "../../src";

test("im2col -> col2im", function () {

  let tensor = Tensor.linspace(1, 16, 16).reshape([1, 1, 4, 4]);
  let x = Learn.constant(tensor);
  let col = Learn.im2col(x, {kernelWidth: 2, kernelHeight: 2, kernelChannel: 1, kernelNum: 1});
  console.log(col.value.toString());

  let im = Learn.col2im(col, {imageHeight: 4, imageWidth: 4, imageNum: 1, imageChannel:1, kernelHeight: 2, kernelWidth: 2});
  console.log(im.value.toString());
});

test("conv2d", function () {

  let image = Tensor.linspace(1, 9, 9).reshape([1, 1, 3, 3]);
  let kernel = Tensor.linspace(1, 4, 4).reshape([1, 1, 2, 2]);
  let x = Learn.parameter(image);
  let y = Learn.parameter(kernel);

  let result = x.conv2d(y, {strideHeight: 1, strideWidth: 1});
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [x, y]);
  let gradX = grads[0];
  let gradY = grads[1];

  console.log(gradX.value.toString());
  console.log(gradY.value.toString());
});