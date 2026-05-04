import { IAgent } from './type';
export declare class AgentsManager {
    private agents;
    /**
     * Register an agent
     * @param agent The agent instance to register
     * @param isDefault Whether to set as default agent
     */
    registerAgent(agent: IAgent): void;
    /**
     * Find agent by name
     * @param name Agent name
     * @returns Found agent instance, undefined if not found
     */
    getAgent(name: string): IAgent | undefined;
    /**
     * Get all registered agents
     * @returns Array of all agent instances
     */
    getAllAgents(): IAgent[];
    /**
     * Get all agent tools
     * @returns Array of tools
     */
    getAllTools(): any[];
}
declare const agentsManager: AgentsManager;
export default agentsManager;
