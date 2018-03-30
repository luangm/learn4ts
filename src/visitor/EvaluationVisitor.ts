import Session from "../Session";
import Add from "../structure/binary/Add";
import Divide from "../structure/binary/Divide";
import MatMul from "../structure/binary/MatMul";
import Maximum from "../structure/binary/Maximum";
import Minimum from "../structure/binary/Minimum";
import Modulo from "../structure/binary/Modulo";
import Multiply from "../structure/binary/Multiply";
import Subtract from "../structure/binary/Subtract";
import Constant from "../structure/core/Constant";
import Parameter from "../structure/core/Parameter";
import Expression from "../structure/Expression";
import ExpressionTypes from "../structure/ExpressionTypes";
import ReduceSum from "../structure/reduction/ReduceSum";
import Assign from "../structure/special/Assign";
import Fill from "../structure/special/Fill";
import Reshape from "../structure/special/Reshape";
import Absolute from "../structure/transform/Absolute";
import Cosh from "../structure/transform/Cosh";
import Cosine from "../structure/transform/Cosine";
import Expm1 from "../structure/transform/Expm1";
import Exponential from "../structure/transform/Exponential";
import Log1p from "../structure/transform/Log1p";
import Logarithm from "../structure/transform/Logarithm";
import Negate from "../structure/transform/Negate";
import Reciprocal from "../structure/transform/Reciprocal";
import ReciprocalGrad from "../structure/transform/ReciprocalGrad";
import Relu from "../structure/transform/Relu";
import Round from "../structure/transform/Round";
import RSqrt from "../structure/transform/RSqrt";
import Sigmoid from "../structure/transform/Sigmoid";
import SigmoidGrad from "../structure/transform/SigmoidGrad";
import Sign from "../structure/transform/Sign";
import Sine from "../structure/transform/Sine";
import Sinh from "../structure/transform/Sinh";
import Softmax from "../structure/transform/Softmax";
import SoftmaxGrad from "../structure/transform/SoftmaxGrad";
import Sqrt from "../structure/transform/Sqrt";
import SqrtGrad from "../structure/transform/SqrtGrad";
import Square from "../structure/transform/Square";
import Step from "../structure/transform/Step";
import Tangent from "../structure/transform/Tangent";
import TangentGrad from "../structure/transform/TangentGrad";
import Tanh from "../structure/transform/Tanh";
import TanhGrad from "../structure/transform/TanhGrad";
import Visitor, {VisitFunc} from "./Visitor";

export default class EvaluationVisitor implements Visitor {

  private _registry: Map<string, VisitFunc>;
  get registry() {
    return this._registry;
  }

  private _session: Session;
  get session() {
    return this._session;
  }

  constructor(session: Session) {
    this._session = session;
    this._registry = new Map<string, VisitFunc>();
    this.init();
  }

  register(type: string, method: VisitFunc): void {
    this.registry.set(type, method);
  }

  visit(node: Expression, params?: any): void {
    if (this.session.isValid(node)) {
      return;
    }

    let method = this.registry.get(node.type);
    if (method) {
      for (let dependency of node.dependencies) {
        dependency.accept(this, params);
      }

      let result = method(node, params);
      this.session.setValue(node, result);
    }
  }

  private init() {
    this.register(ExpressionTypes.Add, Add.evaluate);
    this.register(ExpressionTypes.Divide, Divide.evaluate);
    this.register(ExpressionTypes.MatMul, MatMul.evaluate);
    this.register(ExpressionTypes.Maximum, Maximum.evaluate);
    this.register(ExpressionTypes.Minimum, Minimum.evaluate);
    this.register(ExpressionTypes.Modulo, Modulo.evaluate);
    this.register(ExpressionTypes.Multiply, Multiply.evaluate);
    this.register(ExpressionTypes.Subtract, Subtract.evaluate);

    this.register(ExpressionTypes.Constant, Constant.evaluate);
    this.register(ExpressionTypes.Parameter, Parameter.evaluate);

    this.register(ExpressionTypes.ReduceSum, ReduceSum.evaluate);

    this.register(ExpressionTypes.Assign, Assign.evaluate);
    this.register(ExpressionTypes.Fill, Fill.evaluate);
    // this.register(ExpressionTypes.AddN, AddN.evaluate);
    this.register(ExpressionTypes.Reshape, Reshape.evaluate);

    this.register(ExpressionTypes.Absolute, Absolute.evaluate);
    this.register(ExpressionTypes.Cosine, Cosine.evaluate);
    this.register(ExpressionTypes.Cosh, Cosh.evaluate);
    this.register(ExpressionTypes.Expm1, Expm1.evaluate);
    this.register(ExpressionTypes.Exponential, Exponential.evaluate);
    this.register(ExpressionTypes.Log1p, Log1p.evaluate);
    this.register(ExpressionTypes.Logarithm, Logarithm.evaluate);
    this.register(ExpressionTypes.Negate, Negate.evaluate);
    this.register(ExpressionTypes.Reciprocal, Reciprocal.evaluate);
    this.register(ExpressionTypes.ReciprocalGrad, ReciprocalGrad.evaluate);
    this.register(ExpressionTypes.Relu, Relu.evaluate);
    this.register(ExpressionTypes.Round, Round.evaluate);
    this.register(ExpressionTypes.RSqrt, RSqrt.evaluate);
    this.register(ExpressionTypes.Sigmoid, Sigmoid.evaluate);
    this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.evaluate);
    this.register(ExpressionTypes.Sign, Sign.evaluate);
    this.register(ExpressionTypes.Sine, Sine.evaluate);
    this.register(ExpressionTypes.Sinh, Sinh.evaluate);
    this.register(ExpressionTypes.Softmax, Softmax.evaluate);
    this.register(ExpressionTypes.SoftmaxGrad, SoftmaxGrad.evaluate);
    this.register(ExpressionTypes.Sqrt, Sqrt.evaluate);
    this.register(ExpressionTypes.SqrtGrad, SqrtGrad.evaluate);
    this.register(ExpressionTypes.Square, Square.evaluate);
    this.register(ExpressionTypes.Step, Step.evaluate);
    this.register(ExpressionTypes.Tangent, Tangent.evaluate);
    this.register(ExpressionTypes.TangentGrad, TangentGrad.evaluate);
    this.register(ExpressionTypes.Tanh, Tanh.evaluate);
    this.register(ExpressionTypes.TanhGrad, TanhGrad.evaluate);
  }

}