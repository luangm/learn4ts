import Expression from "./structure/Expression";
import {Tensor} from "tensor4js";
import Add from "./structure/binary/Add";
import Constant from "./structure/core/Constant";
import Graph from "./Graph";
import Session from "./Session";

class Learn4js {

  private _graph: Graph;
  private _interactive: boolean;
  private _session: Session;

  constructor() {
    this._interactive = false;
    this._graph = new Graph("DEFAULT");
    this._session = new Session(this._graph);
    this._graph.session = this._session;
  }

  get activeGraph() {
    return this._graph;
  }

  get activeSession() {
    return this._session;
  }

  add(left: Expression, right: Expression, name?: string): Expression {
    return new Add(left, right, this.activeGraph, name);
  }

  constant(value: Tensor, name?: string): Expression {
    return new Constant(value, this.activeGraph, name);
  }

  create(array: number): Tensor;
  create(array: number[]): Tensor;
  create(array: number[][]): Tensor;
  create(array: number[][][]): Tensor;
  create(array: number[][][][]): Tensor;
  create(array: any): Tensor {
    return Tensor.create(array);
  }

  ones(shape: number[]): Tensor {
    return Tensor.ones(shape);
  }

  scalar(scalar: number): Tensor {
    return Tensor.scalar(scalar);
  }

  zeros(shape: number[]): Tensor {
    return Tensor.zeros(shape);
  }

}

export default new Learn4js();