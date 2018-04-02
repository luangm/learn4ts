import Repeat from "../expression/special/Repeat";
import Session from "../Session";
import Add from "../expression/binary/Add";
import Divide from "../expression/binary/Divide";
import MatMul from "../expression/binary/MatMul";
import Maximum from "../expression/binary/Maximum";
import Minimum from "../expression/binary/Minimum";
import Modulo from "../expression/binary/Modulo";
import Multiply from "../expression/binary/Multiply";
import Subtract from "../expression/binary/Subtract";
import Constant from "../expression/core/Constant";
import Parameter from "../expression/core/Parameter";
import Expression from "../expression/Expression";
import ReduceSum from "../expression/reduction/ReduceSum";
import Assign from "../expression/special/Assign";
import Fill from "../expression/special/Fill";
import Reshape from "../expression/special/Reshape";
import Absolute from "../expression/transform/Absolute";
import Cosh from "../expression/transform/Cosh";
import Cosine from "../expression/transform/Cosine";
import Expm1 from "../expression/transform/Expm1";
import Exponential from "../expression/transform/Exponential";
import Log1p from "../expression/transform/Log1p";
import Logarithm from "../expression/transform/Logarithm";
import Negate from "../expression/transform/Negate";
import Reciprocal from "../expression/transform/Reciprocal";
import ReciprocalGrad from "../expression/transform/ReciprocalGrad";
import Relu from "../expression/transform/Relu";
import Round from "../expression/transform/Round";
import RSqrt from "../expression/transform/RSqrt";
import Sigmoid from "../expression/transform/Sigmoid";
import SigmoidGrad from "../expression/transform/SigmoidGrad";
import Sign from "../expression/transform/Sign";
import Sine from "../expression/transform/Sine";
import Sinh from "../expression/transform/Sinh";
import Softmax from "../expression/transform/Softmax";
import SoftmaxGrad from "../expression/transform/SoftmaxGrad";
import Sqrt from "../expression/transform/Sqrt";
import SqrtGrad from "../expression/transform/SqrtGrad";
import Square from "../expression/transform/Square";
import Step from "../expression/transform/Step";
import Tangent from "../expression/transform/Tangent";
import TangentGrad from "../expression/transform/TangentGrad";
import Tanh from "../expression/transform/Tanh";
import TanhGrad from "../expression/transform/TanhGrad";
import Visitor, {VisitFunc} from "./Visitor";
import {ExpressionTypes} from "../expression/ExpressionTypes";
import Elu from "../expression/transform/Elu";
import Ceil from "../expression/transform/Ceil";
import Floor from "../expression/transform/Floor";
import Softplus from "../expression/transform/Softplus";
import Tile from "../expression/special/Tile";

export default class EvaluationVisitor implements Visitor {

  private readonly _registry: Map<string, VisitFunc>;
  private readonly _session: Session;

  get registry() {
    return this._registry;
  }

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
    this.register(ExpressionTypes.Repeat, Repeat.evaluate);
    this.register(ExpressionTypes.Tile, Tile.evaluate);

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
    this.register(ExpressionTypes.Elu, Elu.evaluate);
    this.register(ExpressionTypes.Round, Round.evaluate);
    this.register(ExpressionTypes.Floor, Floor.evaluate);
    this.register(ExpressionTypes.Ceil, Ceil.evaluate);
    this.register(ExpressionTypes.RSqrt, RSqrt.evaluate);
    this.register(ExpressionTypes.Sigmoid, Sigmoid.evaluate);
    this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.evaluate);
    this.register(ExpressionTypes.Sign, Sign.evaluate);
    this.register(ExpressionTypes.Sine, Sine.evaluate);
    this.register(ExpressionTypes.Sinh, Sinh.evaluate);
    this.register(ExpressionTypes.Softplus, Softplus.evaluate);
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