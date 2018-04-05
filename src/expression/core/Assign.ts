import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Assign extends Expression {

  private readonly _ref: Expression;
  private readonly _source: Expression;

  get dependencies() {
    return [this._source];
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      ref: this.ref.id,
      source: this.source.id
    };
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

  constructor(ref: Expression, source: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._ref = ref;
    this._source = source;
  }

  static evaluate(expression: Expression): undefined {
    let node = expression as Assign;
    node.ref.value = node.source.value;
    return;
  }
}