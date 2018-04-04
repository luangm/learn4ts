import Optimizer from "./Optimizer";
import Graph from "../Graph";
import Expression from "../expression/Expression";
import ReverseGradientVisitor from "../visitor/ReverseGradientVisitor";
import {Tensor} from "tensor4js";

interface GradientDescentOptimizerOptions {
  learnRate?: number;
}

export default class GradientDescentOptimizer implements Optimizer {

  private readonly _graph: Graph;
  private readonly _learnRate: number;
  private readonly _learnRateNode: Expression;

  get graph() {
    return this._graph;
  }

  get learnRate() {
    return this._learnRate;
  }

  constructor(graph: Graph, options: GradientDescentOptimizerOptions) {
    this._graph = graph;
    this._learnRate = options.learnRate || 0.001;
    this._learnRateNode = graph.factory.constant(Tensor.create(this._learnRate), "LearnRate");
  }

  minimize(loss: Expression, params?: Expression[]): Expression {
    // let depVisitor = new DependencyVisitor();
    // loss.accept(depVisitor);
    // let paramNodes = [];
    // for (let node of Object.values(depVisitor.dependencies)) {
    //   if (node instanceof Parameter) {
    //     paramNodes.push(node);
    //   }
    // }

    let paramList: Expression[] = [];
    if (params) {
      paramList = params.slice();
    }

    // Only compute gradients if not already exist.
    if (!loss.hasGradients) {
      let gradVisitor = new ReverseGradientVisitor(this.graph);
      gradVisitor.visit(loss);
    }

    let newValueList: Expression[] = [];
    let assignList: Expression[] = [];
    for (let node of paramList) {
      let grad = loss.getGradient(node);
      // Only update nodes that have a gradient. This handles non-differentiable nodes.
      if (grad) {
        let newVal = node.subtract(grad.multiply(this._learnRateNode)); // node = node - lr * grad
        newValueList.push(newVal);
        let assign = node.assign(newVal);
        assignList.push(assign);
      }
    }

    let groupList = newValueList.concat(assignList);

    return this.graph.factory.group(groupList);
  }

}