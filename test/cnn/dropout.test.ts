import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("dropout", function () {

  let tensor = Tensor.linspace(1, 16, 16).reshape([4, 4]);
  let x = Learn.constant(tensor);

  let z = x.dropout();
  console.log(z.value.toString());
});