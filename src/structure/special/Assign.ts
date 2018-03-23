import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Assign extends Expression {

  private _ref: Expression;
  private _source: Expression;

  constructor(ref: Expression, source: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._ref = ref;
    this._source = source;
  }

  get dependencies() {
    return [this._source];
  }

  get ref() {
    return this._ref;
  }

  get shape() {
    return this._ref.shape;
  }

  get source() {
    return this._source;
  }

  get type() {
    return ExpressionTypes.Assign;
  }

  static evaluate(node: Assign): Tensor {
    let newValue = node.graph.session.getValue(node.source);
    node.graph.session.setValue(node.ref, newValue);
    return newValue;
  }
}