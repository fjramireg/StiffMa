function release()
% release packages the toolbox and stores it in a release folder.
folder = fileparts( mfilename( "fullpath" ) );
tlbxPrj = fullfile( folder, "StiffMa.prj" );
version = matlab.addons.toolbox.toolboxVersion( tlbxPrj );
mltbx = fullfile( folder, "releases", ...
    "StiffMa" + version + ".mltbx" );
matlab.addons.toolbox.packageToolbox( tlbxPrj, mltbx );
end % release
