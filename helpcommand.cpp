/*
 *  helpcommand.cpp
 *  Dotur
 *
 *  Created by Sarah Westcott on 1/2/09.
 *  Copyright 2009 Schloss Lab UMASS Amherst. All rights reserved.
 *
 */

#include "helpcommand.h"

//**********************************************************************************************************************

HelpCommand::HelpCommand(string option){

	
	if (option != "") { mothurOut("There are no valid parameters for the help() command."); mothurOutEndLine();  }
	
	validCommands = new CommandFactory();
}

//**********************************************************************************************************************

HelpCommand::~HelpCommand(){}

//**********************************************************************************************************************

int HelpCommand::execute(){

	validCommands->printCommands(cout);
	mothurOut("For more information about a specific command type 'commandName(help)' i.e. 'read.dist(help)'"); mothurOutEndLine();
	
	delete validCommands;
	
	mothurOutEndLine(); mothurOut("For further assistance please refer to the Mothur manual on our wiki at http://schloss.micro.umass.edu/mothur/, or contact Pat Schloss at pschloss@microbio.umass.edu.\n");
	return 0;
}

//**********************************************************************************************************************/
