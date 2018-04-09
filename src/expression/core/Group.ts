import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Group extends Expression {

  private readonly _list: Expression[];

  get dependencies(): Expression[] {
    return this.list;
  }

  get list() {
    return this._list;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      list: this.list.map(item => item.id) // Note: Order matters
    };
  }

  get shape(): number[] {
    return [];
  }

  get type() {
    return ExpressionTypes.Group;
  }

  constructor(list: Expression[], graph: Graph, name?: string) {
    super(graph, name);
    this._list = list;
  }

}