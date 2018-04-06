import Graph from "../../Graph";
import Expression from "../Expression";
import {Tensor} from "tensor4js";
import {ExpressionTypes} from "../ExpressionTypes";

export default class WhileLoop extends Expression {

  private readonly _body: Expression;
  private readonly _condition: Expression;

  get body() {
    return this._body;
  }

  get condition() {
    return this._condition;
  }

  /**
   * Note: the WhileLoop depends on condition node only.
   */
  get dependencies() {
    return [this._condition];
  }

  get params() {
    return {
      type: this.type,
      name: this.name
    };
  }

  get shape(): number[] {
    return this.body.shape;
  }

  get type(): string {
    return ExpressionTypes.WhileLoop;
  }

  constructor(condition: Expression, body: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._condition = condition;
    this._body = body;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as WhileLoop;
    let condVal = node.condition.value;
    let bodyVal: Tensor;
    while (condVal.data[0]) {
      bodyVal = node.body.value;
      condVal = node.condition.eval();
    }
    return Tensor.scalar(0);
  }

}