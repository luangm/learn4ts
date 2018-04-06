import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("im2col -> col2im", function () {

  let tensor = Tensor.linspace(1, 16, 16).reshape([1, 1, 4, 4]);
  let x = Learn.constant(tensor);
  let col = Learn.im2col(x, {kernelWidth: 2, kernelHeight: 2, kernelChannel: 1, kernelNum: 1});
  console.log(col.value.toString());

  let im = Learn.col2im(col, {imageHeight: 4, imageWidth: 4, imageNum: 1, imageChannel:1, kernelHeight: 2, kernelWidth: 2});
  console.log(im.value.toString());
});