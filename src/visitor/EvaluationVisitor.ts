import Visitor, {VisitFunc} from "./Visitor";
import Expression from "../structure/Expression";
import Session from "../Session";
import Add from "../structure/binary/Add";
import Tensor from "tensor4js/dist/types/Tensor";

export default class EvaluationVisitor implements Visitor {

  private _registry: Map<string, VisitFunc>;
  private _session: Session;

  constructor(session: Session) {
    this._session = session;
    this._registry = new Map<string, VisitFunc>();
    this.init();
  }

  get registry() {
    return this._registry;
  }

  get session() {
    return this._session;
  }

  getValue(node: Expression): Tensor {
    return this.session.getValue(node);
  }

  register(type: string, method: VisitFunc): void {
    this.registry.set(type, method);
  }

  setValue(node: Expression, value: Tensor): void {
    this.session.setValue(node, value);
  }

  visit(node: Expression, params?: any): void {
    let method = this.registry.get(node.type);
    if (method) {
      method(node, this, params);
    }
  }

  private init() {
    this.register(Add.TYPE, Add.evaluate);
  }

}