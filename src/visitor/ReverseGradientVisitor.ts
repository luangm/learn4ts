import Graph from "../Graph";
import Add from "../structure/binary/Add";
import Divide from "../structure/binary/Divide";
import MatMul from "../structure/binary/MatMul";
import Multiply from "../structure/binary/Multiply";
import Subtract from "../structure/binary/Subtract";
import Expression from "../structure/Expression";
import ExpressionTypes from "../structure/ExpressionTypes";
import Absolute from "../structure/transform/Absolute";
import Cosine from "../structure/transform/Cosine";
import Exponential from "../structure/transform/Exponential";
import Logarithm from "../structure/transform/Logarithm";
import Negate from "../structure/transform/Negate";
import Relu from "../structure/transform/Relu";
import Sigmoid from "../structure/transform/Sigmoid";
import Sine from "../structure/transform/Sine";
import Sqrt from "../structure/transform/Sqrt";
import Square from "../structure/transform/Square";
import Tangent from "../structure/transform/Tangent";
import Visitor, {VisitFunc} from "./Visitor";

export default class ReverseGradientVisitor implements Visitor {

  private _gradMap: Map<number, Expression[]>;
  private _graph: Graph;
  private _registry: Map<string, VisitFunc>;
  private _startId: number; // if not 0, the visit already started

  constructor(graph: Graph) {
    this._graph = graph;
    this._registry = new Map<string, VisitFunc>();
    this.init();
  }

  get factory() {
    return this._graph.factory;
  }

  get registry() {
    return this._registry;
  }

  register(type: string, method: VisitFunc): void {
    this.registry.set(type, method);
  }

  visit(node: Expression, params?: any): void {
    // initialize
    if (!this._startId) {
      this._startId = node.id;
      this._gradMap = new Map<number, Expression[]>();
    }

    // body
    let grad = this._graph.addNode(params || this.factory.fill(1, node.shape));
    this.addGradient(node, grad);

    let method = this.registry.get(node.type);
    if (method) {
      let grads = method(node, grad);

      for (let i = 0; i < node.dependencies.length; i++) {
        let dependency = node.dependencies[i];
        dependency.accept(this, grads[i]);
      }
    }

    // finalize
    if (this._startId === node.id) {
      this._startId = 0;
      for (let key of this._gradMap.keys()) {
        let list = this._gradMap.get(key);
        let addN = this.factory.addN(list);
        node.setGradient(key, addN);
      }
      this._gradMap = null;
    }
  }

  private addGradient(target: Expression, grad: Expression) {
    let list = this._gradMap.get(target.id) || [];
    list.push(grad);
    this._gradMap.set(target.id, list);
  }

  private init() {
    this.register(ExpressionTypes.Add, Add.gradients);
    this.register(ExpressionTypes.Divide, Divide.gradients);
    this.register(ExpressionTypes.MatMul, MatMul.gradients);
    // this.register(ExpressionTypes.Maximum, Maximum.gradients);
    // this.register(ExpressionTypes.Minimum, Minimum.gradients);
    // this.register(ExpressionTypes.Modulo, Modulo.gradients);
    this.register(ExpressionTypes.Multiply, Multiply.gradients);
    this.register(ExpressionTypes.Subtract, Subtract.gradients);

    // this.register(ExpressionTypes.Constant, Constant.gradients);
    // this.register(ExpressionTypes.Parameter, Parameter.gradients);

    // this.register(ExpressionTypes.ReduceSum, ReduceSum.gradients);

    // this.register(ExpressionTypes.Assign, Assign.gradients);
    // this.register(ExpressionTypes.Fill, Fill.gradients);

    this.register(ExpressionTypes.Absolute, Absolute.gradients);
    this.register(ExpressionTypes.Cosine, Cosine.gradients);
    // this.register(ExpressionTypes.Expm1, Expm1.gradients);
    this.register(ExpressionTypes.Exponential, Exponential.gradients);
    // this.register(ExpressionTypes.Log1p, Log1p.gradients);
    this.register(ExpressionTypes.Logarithm, Logarithm.gradients);
    this.register(ExpressionTypes.Negate, Negate.gradients);
    // this.register(ExpressionTypes.Reciprocal, Reciprocal.gradients);
    this.register(ExpressionTypes.Relu, Relu.gradients);
    // this.register(ExpressionTypes.Round, Round.gradients);
    // this.register(ExpressionTypes.RSqrt, RSqrt.gradients);
    this.register(ExpressionTypes.Sigmoid, Sigmoid.gradients);
    // this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.gradients);
    // this.register(ExpressionTypes.Sign, Sign.gradients);
    this.register(ExpressionTypes.Sine, Sine.gradients);
    // this.register(ExpressionTypes.Softmax, Softmax.gradients);
    // this.register(ExpressionTypes.SoftmaxGrad, SoftmaxGrad.gradients);
    this.register(ExpressionTypes.Sqrt, Sqrt.gradients);
    // this.register(ExpressionTypes.SqrtGrad, SqrtGrad.gradients);
    this.register(ExpressionTypes.Square, Square.gradients);
    // this.register(ExpressionTypes.Step, Step.gradients);
    this.register(ExpressionTypes.Tangent, Tangent.gradients);
    // this.register(ExpressionTypes.TangentGrad, TangentGrad.gradients);
    // this.register(ExpressionTypes.Tanh, Tanh.gradients);
  }

}