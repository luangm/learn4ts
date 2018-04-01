import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Group extends Expression {

  private _list: Expression[];

  constructor(list: Expression[], graph: Graph, name?: string) {
    super(graph, name);
    this._list = list;
  }

  get dependencies(): Expression[] {
    return this._list;
  }

  get shape(): number[] {
    return [];
  }

  get type() {
    return ExpressionTypes.Group;
  }

}