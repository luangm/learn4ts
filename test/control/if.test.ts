import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("if", function () {

  // let cond = x.greater(y);
  // let sum = x.add(1);
  // let k = Learn.if(x.greater(y), x.add(1))
  //   .elseif(x.less(y), y.subtract(2))
  //   .else(x.multiply(y));

  let x = Learn.parameter(Tensor.create(1));
  let y = Learn.parameter(Tensor.create(2));
  let m = Learn.parameter(Tensor.create(99));
  let n = Learn.parameter(Tensor.create(11));

  let truthy = m.add(n);
  let falsy = m.subtract(n);
  let cond = x.less(y);
  let z = Learn.ifElse(cond, truthy, falsy);

  let s = Learn.constant(Tensor.create(100));
  let k = z.add(s);
  console.log(k.value.toString());
});