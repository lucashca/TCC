import { Coordinates } from "./coordinates";
import { Nutriente } from "./nutriente";

export class Amostra {
  constructor(public name: string, public cord: Coordinates, public elevation: number, public nutrientes: Nutriente[], public icon: string) { }
}
