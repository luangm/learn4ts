import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import {Tensor, TensorMath} from "tensor4js";

export default class Conditional extends Expression {

  private readonly _condition: Expression;
  private readonly _falsy: Expression;
  private readonly _truthy: Expression;

  get condition() {
    return this._condition;
  }

  get dependencies(): Expression[] {
    return [this._condition, this._truthy, this._falsy];
  }

  get falsy() {
    return this._falsy;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      condition: this.condition.id,
      truthy: this.truthy.id,
      falsy: this.falsy.id
    };
  }

  get shape(): number[] {
    return this._condition.shape;
  }

  get truthy() {
    return this._truthy;
  }

  get type() {
    return ExpressionTypes.Conditional;
  }

  constructor(condition: Expression, truthy: Expression, falsy: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._truthy = truthy;
    this._falsy = falsy;
    this._condition = condition;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Conditional;
    let cond = node.condition.value;
    let truthy = node.truthy.value;
    let falsy = node.falsy.value;
    return TensorMath.conditional(cond, truthy, falsy);
  }

}