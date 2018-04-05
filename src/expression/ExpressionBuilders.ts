import Expression from "./Expression";
import Graph from "../Graph";

export class IfElseBuilder {

  private readonly _graph: Graph;
  private readonly _condition: Expression;
  private readonly _then: Expression;
  private _else?: Expression;

  constructor(condition: Expression, then: Expression, graph: Graph) {
    this._graph = graph;
    this._condition = condition;
    this._then = then;
  }

  else(exp: Expression): Expression {
    this._else = exp;
    return this._graph.factory.group([]);
  }
}