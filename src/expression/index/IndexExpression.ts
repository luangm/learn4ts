import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class IndexExpression extends Expression {

  private readonly _base: Expression;
  private readonly _dim: number;
  private readonly _shape: number[];

  get base() {
    return this._base;
  }

  get dependencies() {
    return [this.base];
  }

  get dim() {
    return this._dim;
  }

  get shape() {
    return this._shape;
  }

  protected constructor(base: Expression, dim: number = 0, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._dim = dim;
    this._shape = ShapeUtils.reduceShape(base.shape, dim, false);
  }

}