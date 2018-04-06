import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("while", function () {

  let x = Learn.parameter(Tensor.create(0));
  let y = Learn.parameter(Tensor.create(5));
  let one = Learn.constant(Tensor.create(1));

  let cond = x.less(y);
  let body = y.assign(y.subtract(one));

  let loop = Learn.while(cond, body);

  loop.eval();
});