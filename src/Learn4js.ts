import {Tensor} from "tensor4js";
import Graph from "./Graph";
import Session from "./Session";
import Expression from "./expression/Expression";
import ReverseGradientVisitor from "./visitor/ReverseGradientVisitor";

class Learn4js {

  private readonly _graph: Graph;
  private _interactive: boolean;
  private readonly _session: Session;

  get activeGraph() {
    return this._graph;
  }

  get activeSession() {
    return this._session;
  }

  get factory() {
    return this._graph.factory;
  }

  constructor() {
    this._interactive = false;
    this._graph = new Graph("DEFAULT");
    this._session = new Session(this._graph);
    this._graph.session = this._session;
  }

  abs(base: Expression, name?: string): Expression {
    return this.factory.abs(base, name);
  }

  add(left: Expression, right: Expression, name?: string): Expression {
    return this.factory.add(left, right, name);
  }

  constant(value: Tensor, name?: string): Expression {
    return this.factory.constant(value, name);
  }

  create(array: number): Tensor;

  create(array: number[]): Tensor;

  create(array: number[][]): Tensor;

  create(array: number[][][]): Tensor;

  create(array: number[][][][]): Tensor;

  create(array: any): Tensor {
    return Tensor.create(array);
  }

  gradients(target: Expression, nodes: Expression[]): Expression[] {
    let visitor = new ReverseGradientVisitor(this.activeGraph);
    visitor.visit(target);

    let grads: Expression[] = [];
    for (let node of nodes) {
      let grad = target.getGradient(node);
      if (grad) {
        // if (this.interactive) {
        //   grad.eval();
        // }
        grads.push(grad);
      }
    }
    return grads;
  }

  linspace(start: number, stop: number, num: number): Tensor {
    return Tensor.linspace(start, stop, num);
  }

  negate(base: Expression, name?: string): Expression {
    return this.factory.negate(base, name);
  }

  ones(shape: number[]): Tensor {
    return Tensor.ones(shape);
  }

  parameter(value: Tensor, name?: string): Expression {
    return this.factory.parameter(value, name);
  }

  round(base: Expression, name?: string): Expression {
    return this.factory.round(base, name);
  }

  rsqrt(base: Expression, name?: string): Expression {
    return this.factory.rsqrt(base, name);
  }

  scalar(scalar: number): Tensor {
    return Tensor.scalar(scalar);
  }

  sigmoid(base: Expression, name?: string): Expression {
    return this.factory.sigmoid(base, name);
  }

  sign(base: Expression, name?: string): Expression {
    return this.factory.sign(base, name);
  }

  zeros(shape: number[]): Tensor {
    return Tensor.zeros(shape);
  }
}

export default new Learn4js();