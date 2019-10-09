import { Coordinates } from "./coordinates";
import { Nutriente } from "./nutriente";

export class Amostra {
  constructor(public cord: Coordinates, public nutrientes: Nutriente[]) {}
}
