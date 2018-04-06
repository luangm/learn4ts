import {Tensor} from "tensor4js";
import Graph from "./Graph";
import Session from "./Session";
import Expression from "./expression/Expression";
import ReverseGradientVisitor from "./visitor/ReverseGradientVisitor";
import GradientDescentOptimizer from "./optimizer/GradientDescentOptimizer";
import {Im2ColOptions} from "./expression/nn/Im2Col";
import {Col2ImOptions} from "./expression/nn/Col2Im";

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

  col2im(col: Expression, options: Col2ImOptions, name?: string): Expression {
    return this.factory.col2im(col, options, name);
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

  dropout(base: Expression, probability: number = 0.5, name?: string): Expression {
    return this.factory.dropout(base, probability, name);
  }

  gradientDescent(learnRate: number = 0.001) {
    return new GradientDescentOptimizer(this.activeGraph, {learnRate});
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

  group(list: Expression[], name?: string): Expression {
    return this.factory.group(list, name);
  }

  ifElse(condition: Expression, truthy: Expression, falsy: Expression, name?: string): Expression {
    return this.factory.ifElse(condition, truthy, falsy, name);
  }

  im2col(image: Expression, options: Im2ColOptions, name?: string): Expression {
    return this.factory.im2col(image, options, name);
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

  sparseZeros(shape: number[]): Tensor {
    return Tensor.sparseZeros(shape);
  }

  variable(shape: number[], name?: string): Expression {
    return this.factory.variable(shape, name);
  }

  while(condition: Expression, body: Expression, name?: string): Expression {
    return this.factory.while(condition, body, name);
  }

  zeros(shape: number[]): Tensor {
    return Tensor.zeros(shape);
  }
}

export default new Learn4js();