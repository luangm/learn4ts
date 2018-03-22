import Expression from "../structure/Expression";

export type VisitFunc = (node: Expression, visitor: Visitor, params?: any) => void;

export default interface Visitor {

  readonly registry: Map<string, VisitFunc>;

  register(type: string, method: VisitFunc): void;

  visit(node: Expression, params?: any): void;
}