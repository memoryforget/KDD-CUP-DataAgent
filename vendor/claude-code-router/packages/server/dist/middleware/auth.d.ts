import { FastifyRequest, FastifyReply } from "fastify";
export declare const apiKeyAuth: (config: any) => (req: FastifyRequest, reply: FastifyReply, done: () => void) => Promise<void>;
