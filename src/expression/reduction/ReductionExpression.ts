import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class ReductionExpression extends Expression {

  private readonly _base: Expression;
  private readonly _dims: number | number[];
  private readonly _shape: number[];

  get base() {
    return this._base;
  }

  get dependencies() {
    return [this.base];
  }

  get dims() {
    return this._dims;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      dims: this.dims
    };
  }

  get shape() {
    return this._shape;
  }

  protected constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._dims = dims;
    this._shape = ShapeUtils.reduceShape(base.shape, dims, false);
  }

}