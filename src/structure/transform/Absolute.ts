import TransformExpression from "./TransformExpression";
import Expression from "../Expression";
import Graph from "../../Graph";

export default class Absolute extends TransformExpression {

  static TYPE = "Absolute";

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return Absolute.TYPE;
  }

}