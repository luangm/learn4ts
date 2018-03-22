import Graph from "./Graph";
import Expression from "./structure/Expression";
import {Tensor} from "tensor4js";

export default class Session {

  constructor(graph: Graph) {

  }

  getValue(node: Expression): Tensor {
    return Tensor.ones(node.shape);
  }

  isValid(node: Expression): boolean {
    return false;
  }

  setValue(node: Expression, value: Tensor): void {
    // this.session.setValue(node, value);
  }
}