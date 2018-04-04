import Graph from "../Graph";
import Expression from "../expression/Expression";

export default interface Optimizer {

  /**
   * readonly reference to graph
   */
  readonly graph: Graph;

  /**
   * Returns a pointer to minimization steps.
   * Optionally specify a list of Parameters to train on.
   * If params is not specified, then all dependent params are used.
   */
  minimize(loss: Expression, params?: Expression[]): Expression;

}