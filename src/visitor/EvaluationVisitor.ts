import Add from "../expression/binary/Add";
import Divide from "../expression/binary/Divide";
import FloorDiv from "../expression/binary/FloorDiv";
import FloorMod from "../expression/binary/FloorMod";
import MatMul from "../expression/binary/MatMul";
import Maximum from "../expression/binary/Maximum";
import Minimum from "../expression/binary/Minimum";
import Multiply from "../expression/binary/Multiply";
import Power from "../expression/binary/Power";
import Subtract from "../expression/binary/Subtract";
import TruncateDiv from "../expression/binary/TruncateDiv";
import TruncateMod from "../expression/binary/TruncateMod";
import Equal from "../expression/comparison/Equal";
import Greater from "../expression/comparison/Greater";
import GreaterEqual from "../expression/comparison/GreaterEqual";
import Less from "../expression/comparison/Less";
import LessEqual from "../expression/comparison/LessEqual";
import NotEqual from "../expression/comparison/NotEqual";
import IfElse from "../expression/control/IfElse";
import WhileLoop from "../expression/control/WhileLoop";
import Assign from "../expression/core/Assign";
import Constant from "../expression/core/Constant";
import Parameter from "../expression/core/Parameter";
import Zeros from "../expression/core/Zeros";
import Expression from "../expression/Expression";
import {ExpressionTypes} from "../expression/ExpressionTypes";
import ArgMax from "../expression/index/ArgMax";
import ArgMin from "../expression/index/ArgMin";
import Col2Im from "../expression/nn/Col2Im";
import Dropout from "../expression/nn/Dropout";
import Im2Col from "../expression/nn/Im2Col";
import ReduceLogSumExp from "../expression/reduction/ReduceLogSumExp";
import ReduceMax from "../expression/reduction/ReduceMax";
import ReduceMean from "../expression/reduction/ReduceMean";
import ReduceMin from "../expression/reduction/ReduceMin";
import ReduceProd from "../expression/reduction/ReduceProd";
import ReduceSum from "../expression/reduction/ReduceSum";
import AddN from "../expression/special/AddN";
import Conditional from "../expression/special/Conditional";
import Fill from "../expression/special/Fill";
import Repeat from "../expression/special/Repeat";
import Reshape from "../expression/special/Reshape";
import Slice from "../expression/special/Slice";
import Tile from "../expression/special/Tile";
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
import EluGrad from "../expression/transform/EluGrad";
import Erf from "../expression/transform/Erf";
import Erfc from "../expression/transform/Erfc";
import ErfcGrad from "../expression/transform/ErfcGrad";
import ErfGrad from "../expression/transform/ErfGrad";
import Expm1 from "../expression/transform/Expm1";
import Exponential from "../expression/transform/Exponential";
import Floor from "../expression/transform/Floor";
import Gamma from "../expression/transform/Gamma";
import LGamma from "../expression/transform/LGamma";
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
import Softplus from "../expression/transform/Softplus";
import Sqrt from "../expression/transform/Sqrt";
import SqrtGrad from "../expression/transform/SqrtGrad";
import Square from "../expression/transform/Square";
import Step from "../expression/transform/Step";
import Tangent from "../expression/transform/Tangent";
import TangentGrad from "../expression/transform/TangentGrad";
import Tanh from "../expression/transform/Tanh";
import TanhGrad from "../expression/transform/TanhGrad";
import Session from "../Session";
import Visitor, {VisitFunc} from "./Visitor";

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

    if (node.internal) {
      node.internal.accept(this, params);
      let result = this.session.getValue(node.internal);
      if (result) {
        this.session.setValue(node, result);
      }
      return;
    }

    for (let dependency of node.dependencies) {
      dependency.accept(this, params);
    }

    let method = this.registry.get(node.type);
    if (method) {
      let result = method(node, params);
      if (result) {
        // console.log(node.type, result.toString());
        this.session.setValue(node, result);
      }
    } else {
      console.warn("No evaluate method for node: " + node.type);
    }

  }

  private init() {
    this.register(ExpressionTypes.Add, Add.evaluate);
    this.register(ExpressionTypes.Divide, Divide.evaluate);
    this.register(ExpressionTypes.MatMul, MatMul.evaluate);
    this.register(ExpressionTypes.Maximum, Maximum.evaluate);
    this.register(ExpressionTypes.Minimum, Minimum.evaluate);
    this.register(ExpressionTypes.FloorMod, FloorMod.evaluate);
    this.register(ExpressionTypes.FloorDiv, FloorDiv.evaluate);
    this.register(ExpressionTypes.TruncateMod, TruncateMod.evaluate);
    this.register(ExpressionTypes.TruncateDiv, TruncateDiv.evaluate);
    this.register(ExpressionTypes.Multiply, Multiply.evaluate);
    this.register(ExpressionTypes.Subtract, Subtract.evaluate);
    this.register(ExpressionTypes.Power, Power.evaluate);

    this.register(ExpressionTypes.Constant, Constant.evaluate);
    this.register(ExpressionTypes.Parameter, Parameter.evaluate);
    this.register(ExpressionTypes.Zeros, Zeros.evaluate);
    // this.register(ExpressionTypes.Group, Group.evaluate);

    this.register(ExpressionTypes.ReduceSum, ReduceSum.evaluate);
    this.register(ExpressionTypes.ReduceMean, ReduceMean.evaluate);
    this.register(ExpressionTypes.ReduceMax, ReduceMax.evaluate);
    this.register(ExpressionTypes.ReduceMin, ReduceMin.evaluate);
    this.register(ExpressionTypes.ReduceProd, ReduceProd.evaluate);
    this.register(ExpressionTypes.ReduceLogSumExp, ReduceLogSumExp.evaluate);

    this.register(ExpressionTypes.Assign, Assign.evaluate);
    this.register(ExpressionTypes.Fill, Fill.evaluate);
    this.register(ExpressionTypes.AddN, AddN.evaluate);
    this.register(ExpressionTypes.Reshape, Reshape.evaluate);
    this.register(ExpressionTypes.Repeat, Repeat.evaluate);
    this.register(ExpressionTypes.Tile, Tile.evaluate);
    this.register(ExpressionTypes.Transpose, Transpose.evaluate);
    this.register(ExpressionTypes.Slice, Slice.evaluate);

    this.register(ExpressionTypes.Absolute, Absolute.evaluate);
    this.register(ExpressionTypes.Expm1, Expm1.evaluate);
    this.register(ExpressionTypes.Exponential, Exponential.evaluate);
    this.register(ExpressionTypes.Log1p, Log1p.evaluate);
    this.register(ExpressionTypes.Logarithm, Logarithm.evaluate);
    this.register(ExpressionTypes.Duplicate, Duplicate.evaluate);
    this.register(ExpressionTypes.Negate, Negate.evaluate);
    this.register(ExpressionTypes.Reciprocal, Reciprocal.evaluate);
    this.register(ExpressionTypes.ReciprocalGrad, ReciprocalGrad.evaluate);
    this.register(ExpressionTypes.Relu, Relu.evaluate);
    this.register(ExpressionTypes.Elu, Elu.evaluate);
    this.register(ExpressionTypes.EluGrad, EluGrad.evaluate);
    this.register(ExpressionTypes.Round, Round.evaluate);
    this.register(ExpressionTypes.Floor, Floor.evaluate);
    this.register(ExpressionTypes.Ceil, Ceil.evaluate);
    this.register(ExpressionTypes.RSqrt, RSqrt.evaluate);
    this.register(ExpressionTypes.Sigmoid, Sigmoid.evaluate);
    this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.evaluate);
    this.register(ExpressionTypes.Sign, Sign.evaluate);
    this.register(ExpressionTypes.Softplus, Softplus.evaluate);
    this.register(ExpressionTypes.Softmax, Softmax.evaluate);
    this.register(ExpressionTypes.SoftmaxGrad, SoftmaxGrad.evaluate);
    this.register(ExpressionTypes.Sqrt, Sqrt.evaluate);
    this.register(ExpressionTypes.SqrtGrad, SqrtGrad.evaluate);
    this.register(ExpressionTypes.Square, Square.evaluate);
    this.register(ExpressionTypes.Step, Step.evaluate);

    this.register(ExpressionTypes.Equal, Equal.evaluate);
    this.register(ExpressionTypes.NotEqual, NotEqual.evaluate);
    this.register(ExpressionTypes.Greater, Greater.evaluate);
    this.register(ExpressionTypes.GreaterEqual, GreaterEqual.evaluate);
    this.register(ExpressionTypes.Less, Less.evaluate);
    this.register(ExpressionTypes.LessEqual, LessEqual.evaluate);
    this.register(ExpressionTypes.Conditional, Conditional.evaluate);

    this.register(ExpressionTypes.Sine, Sine.evaluate);
    this.register(ExpressionTypes.Sinh, Sinh.evaluate);
    this.register(ExpressionTypes.Cosine, Cosine.evaluate);
    this.register(ExpressionTypes.Cosh, Cosh.evaluate);
    this.register(ExpressionTypes.Tangent, Tangent.evaluate);
    this.register(ExpressionTypes.TangentGrad, TangentGrad.evaluate);
    this.register(ExpressionTypes.Tanh, Tanh.evaluate);
    this.register(ExpressionTypes.TanhGrad, TanhGrad.evaluate);
    this.register(ExpressionTypes.Asin, Asin.evaluate);
    this.register(ExpressionTypes.Asinh, Asinh.evaluate);
    this.register(ExpressionTypes.Acos, Acos.evaluate);
    this.register(ExpressionTypes.Acosh, Acosh.evaluate);
    this.register(ExpressionTypes.Atan, Atan.evaluate);
    this.register(ExpressionTypes.Atanh, Atanh.evaluate);

    this.register(ExpressionTypes.ArgMax, ArgMax.evaluate);
    this.register(ExpressionTypes.ArgMin, ArgMin.evaluate);

    this.register(ExpressionTypes.IfElse, IfElse.evaluate);
    this.register(ExpressionTypes.WhileLoop, WhileLoop.evaluate);

    this.register(ExpressionTypes.Im2Col, Im2Col.evaluate);
    this.register(ExpressionTypes.Col2Im, Col2Im.evaluate);
    this.register(ExpressionTypes.Dropout, Dropout.evaluate);
    // this.register(ExpressionTypes.Conv2d, Conv2d.evaluate);

    this.register(ExpressionTypes.Erf, Erf.evaluate);
    this.register(ExpressionTypes.Erfc, Erfc.evaluate);
    this.register(ExpressionTypes.ErfGrad, ErfGrad.evaluate);
    this.register(ExpressionTypes.ErfcGrad, ErfcGrad.evaluate);
    this.register(ExpressionTypes.Gamma, Gamma.evaluate);
    this.register(ExpressionTypes.LGamma, LGamma.evaluate);

  }

}