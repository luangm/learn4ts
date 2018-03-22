import Expression from "../Expression";
import {Tensor} from "tensor4js";
import Graph from "../../Graph";

export default class Constant extends Expression {

  static TYPE = "Constant";

  private _value: Tensor;

  constructor(value: Tensor, graph: Graph, name?: string) {
    super(graph, name);
    this._value = value;
  }

  get shape() {
    return this._value.shape;
  }

  get type() {
    return Constant.TYPE;
  }

  get value() {
    return this._value;
  }

  set value(val) {
    throw new Error('Cannot set constant');
  }

}