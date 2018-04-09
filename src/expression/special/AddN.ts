import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

/**
 * AddN node is trying to add up multiple nodes at the same time.
 * this is designed for adding up gradients, not for normal graphs.
 * This op does NOT broadcast. All nodes must have the same shape.
 */
export default class AddN extends Expression {

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
      list: this.list.map(item => item.id).sort() // Note: Order DOES NOT matter
    };
  }

  get shape() {
    return this._list[0].shape;
  }

  get type() {
    return ExpressionTypes.AddN;
  }

  constructor(list: Expression[], graph: Graph, name?: string) {
    super(graph, name);
    this._list = list;
  }

}