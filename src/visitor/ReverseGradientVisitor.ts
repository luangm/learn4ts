import Add from "../expression/binary/Add";
import Divide from "../expression/binary/Divide";
import FloorMod from "../expression/binary/FloorMod";
import MatMul from "../expression/binary/MatMul";
import Maximum from "../expression/binary/Maximum";
import Minimum from "../expression/binary/Minimum";
import Multiply from "../expression/binary/Multiply";
import Power from "../expression/binary/Power";
import Subtract from "../expression/binary/Subtract";
import Expression from "../expression/Expression";
import {ExpressionTypes} from "../expression/ExpressionTypes";
import Conv2d from "../expression/nn/Conv2d";
import ReduceLogSumExp from "../expression/reduction/ReduceLogSumExp";
import ReduceMax from "../expression/reduction/ReduceMax";
import ReduceMean from "../expression/reduction/ReduceMean";
import ReduceMin from "../expression/reduction/ReduceMin";
import ReduceSum from "../expression/reduction/ReduceSum";
import AddN from "../expression/special/AddN";
import Reshape from "../expression/special/Reshape";
import Transpose from "../expression/special/Transpose";
import Absolute from "../expression/transform/Absolute";
import Acos from "../expression/transform/Acos";
import Acosh from "../expression/transform/Acosh";
import Asin from "../expression/transform/Asin";
import Asinh from "../expression/transform/Asinh";
import Atan from "../expression/transform/Atan";
import Atanh from "../expression/transform/Atanh";
import Ceil from "../expression/transform/Ceil";
import Cosh from "../expression/transform/Cosh";
import Cosine from "../expression/transform/Cosine";
import Duplicate from "../expression/transform/Duplicate";
import Elu from "../expression/transform/Elu";
import Erf from "../expression/transform/Erf";
import Erfc from "../expression/transform/Erfc";
import Expm1 from "../expression/transform/Expm1";
import Exponential from "../expression/transform/Exponential";
import Floor from "../expression/transform/Floor";
import Log1p from "../expression/transform/Log1p";
import Logarithm from "../expression/transform/Logarithm";
import Negate from "../expression/transform/Negate";
import Reciprocal from "../expression/transform/Reciprocal";
import Relu from "../expression/transform/Relu";
import Round from "../expression/transform/Round";
import Sigmoid from "../expression/transform/Sigmoid";
import Sign from "../expression/transform/Sign";
import Sine from "../expression/transform/Sine";
import Sinh from "../expression/transform/Sinh";
import Softmax from "../expression/transform/Softmax";
import Softplus from "../expression/transform/Softplus";
import Sqrt from "../expression/transform/Sqrt";
import Square from "../expression/transform/Square";
import Step from "../expression/transform/Step";
import Tangent from "../expression/transform/Tangent";
import Tanh from "../expression/transform/Tanh";
import Graph from "../Graph";
import Visitor, {VisitFunc} from "./Visitor";

export default class ReverseGradientVisitor implements Visitor {

  private _gradMap: Map<number, Expression[]>;
  private readonly _graph: Graph;
  private readonly _registry: Map<string, VisitFunc>;
  private _startId: number = 0; // if not 0, the visit already started

  get factory() {
    return this.graph.factory;
  }

  get graph() {
    return this._graph;
  }

  get registry() {
    return this._registry;
  }

  constructor(graph: Graph) {
    this._graph = graph;
    this._registry = new Map<string, VisitFunc>();
    this._gradMap = new Map<number, Expression[]>();
    this.init();
  }

  register(type: string, method: VisitFunc): void {
    this.registry.set(type, method);
  }

  visit(node: Expression, params?: any): void {
    // initialize
    if (this._startId === 0) {
      this._startId = node.id;
      this._gradMap = new Map<number, Expression[]>();
    }

    // body
    let grad = params || this.factory.fill(1, node.shape);
    this.addGradient(node, grad);

    // Check if a gradient method is registered for this type of node
    let method = this.registry.get(node.type);
    if (method) {
      // found a method, use this method
      let grads = method(node, grad);
      for (let i = 0; i < node.dependencies.length; i++) {
        let dependency = node.dependencies[i];
        dependency.accept(this, grads[i]);
      }
    } else if (node.internal) {
      // Not found a method,
      // Has an internal node, then only need to build internal node's gradients
      node.internal.accept(this, grad);
    } else if (node.notDifferentiable) {
      // Node is not differentiable, do nothing.
    } else {
      console.warn("Gradient is not defined for node type of " + node.type);
      // throw new Error();
    }

    // finalize
    if (this._startId === node.id) {
      this._startId = 0;
      for (let key of this._gradMap.keys()) {
        let list = this._gradMap.get(key);
        if (list) {
          let addN = this.factory.addN(list);
          node.setGradient(key, addN);
        }
      }
      this._gradMap.clear();
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
    this.register(ExpressionTypes.Maximum, Maximum.gradients);
    this.register(ExpressionTypes.Minimum, Minimum.gradients);
    this.register(ExpressionTypes.FloorMod, FloorMod.gradients);
    this.register(ExpressionTypes.Multiply, Multiply.gradients);
    this.register(ExpressionTypes.Subtract, Subtract.gradients);
    this.register(ExpressionTypes.Power, Power.gradients);
    this.register(ExpressionTypes.Softmax, Softmax.gradients);

    this.register(ExpressionTypes.ReduceSum, ReduceSum.gradients);
    this.register(ExpressionTypes.ReduceMean, ReduceMean.gradients);
    this.register(ExpressionTypes.ReduceMax, ReduceMax.gradients);
    this.register(ExpressionTypes.ReduceMin, ReduceMin.gradients);
    this.register(ExpressionTypes.ReduceLogSumExp, ReduceLogSumExp.gradients);
    // this.register(ExpressionTypes.Assign, Assign.gradients);
    // this.register(ExpressionTypes.Fill, Fill.gradients);

    this.register(ExpressionTypes.Absolute, Absolute.gradients);
    this.register(ExpressionTypes.Duplicate, Duplicate.gradients);
    this.register(ExpressionTypes.Expm1, Expm1.gradients);
    this.register(ExpressionTypes.Exponential, Exponential.gradients);
    this.register(ExpressionTypes.Log1p, Log1p.gradients);
    this.register(ExpressionTypes.Logarithm, Logarithm.gradients);
    this.register(ExpressionTypes.Negate, Negate.gradients);
    this.register(ExpressionTypes.Reciprocal, Reciprocal.gradients);
    this.register(ExpressionTypes.Relu, Relu.gradients);
    this.register(ExpressionTypes.Elu, Elu.gradients);
    // this.register(ExpressionTypes.Round, Round.gradients);
    // this.register(ExpressionTypes.RSqrt, RSqrt.gradients);
    this.register(ExpressionTypes.Sigmoid, Sigmoid.gradients);
    // this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.gradients);
    this.register(ExpressionTypes.Sign, Sign.gradients);

    this.register(ExpressionTypes.Softplus, Softplus.gradients);
    // this.register(ExpressionTypes.Softmax, Softmax.gradients);
    // this.register(ExpressionTypes.SoftmaxGrad, SoftmaxGrad.gradients);
    this.register(ExpressionTypes.Sqrt, Sqrt.gradients);
    // this.register(ExpressionTypes.SqrtGrad, SqrtGrad.gradients);
    this.register(ExpressionTypes.Square, Square.gradients);
    this.register(ExpressionTypes.Step, Step.gradients);

    // this.register(ExpressionTypes.TangentGrad, TangentGrad.gradients);

    this.register(ExpressionTypes.Reshape, Reshape.gradients);
    this.register(ExpressionTypes.Floor, Floor.gradients);
    this.register(ExpressionTypes.Ceil, Ceil.gradients);
    this.register(ExpressionTypes.Round, Round.gradients);
    this.register(ExpressionTypes.Transpose, Transpose.gradients);

    this.register(ExpressionTypes.Sine, Sine.gradients);
    this.register(ExpressionTypes.Sinh, Sinh.gradients);
    this.register(ExpressionTypes.Asin, Asin.gradients);
    this.register(ExpressionTypes.Asinh, Asinh.gradients);
    this.register(ExpressionTypes.Cosine, Cosine.gradients);
    this.register(ExpressionTypes.Cosh, Cosh.gradients);
    this.register(ExpressionTypes.Acos, Acos.gradients);
    this.register(ExpressionTypes.Acosh, Acosh.gradients);
    this.register(ExpressionTypes.Tangent, Tangent.gradients);
    this.register(ExpressionTypes.Tanh, Tanh.gradients);
    this.register(ExpressionTypes.Atan, Atan.gradients);
    this.register(ExpressionTypes.Atanh, Atanh.gradients);

    this.register(ExpressionTypes.Erf, Erf.gradients);
    this.register(ExpressionTypes.Erfc, Erfc.gradients);

    this.register(ExpressionTypes.Conv2d, Conv2d.gradients);

    this.register(ExpressionTypes.AddN, AddN.gradients);
  }

}